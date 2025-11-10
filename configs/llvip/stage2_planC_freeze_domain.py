"""Stage2 KAIST - Plan C: Freeze Domain Alignment (6 epochs)

应急方案：如果方案A在epoch 4-6时mAP<0.55，立即切换到此配置

策略：
1. 完全关闭域对齐（domain_weight=0.0, enable_mmd=False）
2. 专注优化检测性能
3. 从epoch_3.pth或best_checkpoint恢复训练
4. 训练6 epochs后评估，如果mAP>0.65再启用域对齐

预期：
- 目标mAP: 0.65-0.68 @ epoch 6 (相当于累计9 epochs)
- 之后可以考虑Phase 2精调（启用域对齐）
"""

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *  # noqa

# Model definition - FREEZE DOMAIN ALIGNMENT
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
            domain_weight=0.0,  # ⚠️ FROZEN: 完全关闭域对齐
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
                loss_weight=0.0  # ⚠️ FROZEN: MMD loss权重也设为0
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

val_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_evaluator = val_evaluator

# Training schedule - 6 epochs for Phase 1
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=6,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - 保持2.5e-4学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2.5e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        }
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

fp16 = dict(loss_scale='dynamic')

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.5e-5,
        begin=1,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Hooks - 移除DomainWeightWarmupHook
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='pascal_voc/mAP',
        rule='greater',
        max_keep_ckpts=5
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(type='TSNEVisualHook', interval=1, num_samples=200)
    # ⚠️ 注意：移除了DomainWeightWarmupHook
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

# ⚠️ 从方案A的checkpoint恢复（手动设置为epoch_3.pth或best checkpoint）
# 需要根据实际情况修改这个路径
load_from = 'work_dirs/stage2_kaist_full_conservative/best_pascal_voc_mAP_epoch_3.pth'
resume = False

launcher = 'none'
work_dir = './work_dirs/stage2_planC_freeze_domain'
