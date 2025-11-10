"""Stage2 KAIST Finetune with Domain Alignment + MACL + MSP + DHN

Key settings:
    - Load Stage1 checkpoint (epoch_21.pth)
    - Freeze backbone stages (frozen_stages=2)
    - Enable domain alignment (MMD) inside MACLHead (lambda=0.1 warmup)
    - AdamW optimizer (lr=3e-4) + Linear warmup (400 iters) + Cosine 12 epochs
    - Batch size 4, fp16 enabled
    - TSNEVisualHook + (optional) fixed DHN sampler (top_k=128)
    - KAISTDataset (VOC style) for train/val/test (person-only)

Domain loss warmup: first 2 epochs linearly scale domain_weight from 0 -> 0.1
Implemented via simple ParamScheduler-like hook override (custom hook below).
"""

# Explicitly import custom modules to ensure registry
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.models.macldhnmsp',
        'mmdet.models.roi_heads.aligned_roi_head',
        'mmdet.models.utils.domain_aligner',
        'mmdet.engine.hooks.domain_weight_warmup_hook',
        'mmdet.datasets.kaist'
    ],
    allow_failed_imports=False
)

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *  # noqa

# Model definition (explicit for clarity, based on Faster R-CNN R50-FPN)
model = dict(
    type='FasterRCNN',
    # Use PairedDetDataPreprocessor for unified visible+infrared preprocessing
    data_preprocessor=dict(
        type='PairedDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=True  # Ensure gt_bboxes are converted to Tensor for consistent types
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256,512,1024,2048],
        out_channels=256,
        num_outs=5,
        use_msp=True,
        msp_module=dict(type='MSPReweight', channels=256)
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]
        ),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[1.,1.,1.,1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4,8,16,32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
        # MACL + Domain Alignment + DHN
        use_macl=True,
        use_msp=True,
        use_dhn=True,
        macl_head=dict(
            type='MACLHead',
            in_dim=256,
            proj_dim=128,
            hidden_dims=(256,128),
            norm_type='GN',
            norm_groups=32,
            dropout=0.1,
            use_dhn=True,
            dhn_cfg=dict(queue_size=8192, momentum=0.99, top_k=128),
            enable_domain_loss=True,
            domain_weight=0.1,  # will be warmed from 0
            mmd_kernels=(1.0,2.0,4.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False
        ),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
    )
)

# Optimizer & FP16
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=5e-4),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

fp16 = dict(loss_scale='dynamic')

# Schedulers: 400 iters warmup (approx) then cosine 12 epochs
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=400),
    dict(type='CosineAnnealingLR', T_max=12, begin=0, end=12, eta_min=1e-6, by_epoch=True)
]

# Hooks (keep TSNE + optional schedule placeholder)
custom_hooks = [
    dict(type='TSNEVisualHook', interval=1, num_samples=200),
    # 域对齐权重热身：前2个epoch从0线性升至0.1
    dict(
        type='DomainWeightWarmupHook',
        attr_path='roi_head.macl_head.domain_weight',
        start=0.0,
        target=0.1,
        warmup_epochs=2,
        mode='linear',
        verbose=True,
    ),
]

# Data pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_subdir='Annotations',
        ann_file='C:/KAIST_processed/ImageSets/train.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        return_modality_pair=True,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_subdir='Annotations',
        ann_file='C:/KAIST_processed/ImageSets/val.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        return_modality_pair=False,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='pascal_voc/mAP', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Load Stage1 best checkpoint
load_from = 'work_dirs/stage1_longrun_full/epoch_21.pth'

work_dir = './work_dirs/stage2_kaist_finetune'

# Randomness control (optional reproducibility)
randomness = dict(seed=42, deterministic=False)
