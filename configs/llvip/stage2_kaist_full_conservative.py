"""Stage2 KAIST Full Training - Conservative Strategy (12 epochs)

优化策略（针对sanity run的mAP下降问题）：
1. 降低学习率：3e-4 → 2.5e-4（与Stage1一致）
2. 延长warmup：400 iters → 500 iters
3. 温和domain weight：target 0.1 → 0.08, warmup 2 → 4 epochs
4. 保存更多checkpoint：max_keep_ckpts=5

预期性能：
- 目标mAP: 0.68-0.72 @ epoch 12
- 训练时间: ~30小时 (12 epochs × 2.5h/epoch)
- 内存占用: 2145 MB训练, 740 MB验证（已验证稳定）

对比sanity run问题：
- Sanity run: epoch1 62.8% → epoch3 55.0% (下降12.4%)
- 分析：domain_weight快速增长(0.05→0.1)导致短期检测性能牺牲
- 解决：更低target(0.08) + 更长warmup(4 epochs) = 平滑过渡
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

# Model definition
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='PairedDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=True
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
            scales=[8],
            ratios=[0.5,1.0,2.0],
            strides=[4,8,16,32,64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.,0.,0.,0.],
            target_stds=[1.0,1.0,1.0,1.0]
        ),
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
            type='MACLHead',
            enable_macl=True,
            enable_msp=True,
            enable_dhn=True,
            enable_domain_align=False,
            domain_weight=0.08,  # 降低到0.08（sanity run中0.1导致mAP下降）
            with_avg_pool=True,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.,0.,0.,0.],
                target_stds=[0.1,0.1,0.2,0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            macl_cfg=dict(
                temperature=0.1,
                loss_weight=0.5,
                num_hard_negatives=64
            ),
            dhn_cfg=dict(
                top_k=128,
                num_bins=3,
                loss_weight=0.3
            ),
            mmd_cfg=dict(
                kernel='rbf',
                num_layers=2,
                loss_weight=0.1
            )
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

# Dataset settings
dataset_type = 'KAISTDataset'
data_root = 'data/kaist-rgbt/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        return_modality_pair=True,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_modality_pair=True,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_evaluator = val_evaluator

# Training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - 降低学习率到2.5e-4（与Stage1一致）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2.5e-4, weight_decay=0.01),  # 从3e-4降低
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone': dict(lr_mult=0.1)  # backbone更低学习率
        }
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# FP16 training
fp16 = dict(loss_scale='dynamic')

# Learning rate schedule - 延长warmup到500 iters
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500  # 从400增加到500 iters
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.5e-5,  # 最小lr = 初始lr的10%
        begin=1,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='pascal_voc/mAP',
        rule='greater',
        max_keep_ckpts=5  # 保存top-5模型
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(type='TSNEVisualHook', interval=1, num_samples=200),
    # 更温和的domain weight warmup
    dict(
        type='DomainWeightWarmupHook',
        attr_path='roi_head.macl_head.domain_weight',
        start=0.0,
        target=0.08,  # 从0.1降低到0.08
        warmup_epochs=4,  # 从2增加到4 epochs
        mode='linear',
        verbose=True
    )
]

# Runtime settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'work_dirs/stage1_longrun_full/epoch_21.pth'
resume = False

# Launcher
launcher = 'none'
work_dir = './work_dirs/stage2_kaist_full_conservative'
