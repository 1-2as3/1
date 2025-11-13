"""
Plan B: MACL Progressive Rescue Configuration
==============================================
目标: 用极低学习率 + 轻量级MACL (lambda1=0.005) 重建跨域语义一致性
触发条件: Resume训练mAP持续下降至0.53 (< 0.55阈值)

核心策略:
1. 超低学习率 (lr=1e-5, backbone=5e-7) - 避免特征空间震荡
2. 轻量级MACL (lambda1=0.005, 仅1/10原始权重) - 渐进式语义对齐
3. 禁用DHN和Domain Confusion - 专注语义一致性
4. 渐进式Lambda增长 (每2个epoch: 0.005→0.0075→0.01125)
5. 从最佳Epoch 2 checkpoint恢复 (mAP=0.5884)

预期效果:
- Epoch 1-2: 稳定特征空间,mAP回升至0.56+
- Epoch 3-4: MACL权重增加,mAP突破0.60
- Epoch 5-6: 达到或超越原始Stage2.1目标 (mAP≥0.63)
"""

# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py',
#     '../_base_/datasets/voc0712.py',
#     '../_base_/default_runtime.py'
# ]
# 注释掉base imports,使用完整配置

# ============================================================================
# Model Configuration - MACL Rescue Mode
# ============================================================================
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',  # 使用你的自定义RoIHead (支持MACL)
        
        # ========== MACL Rescue Configuration ==========
        use_macl=True,           # 重新启用MACL
        macl_head=dict(          # MACL Head配置
            type='MACLHead',
            in_dim=1024,         # 修正: 不是in_channels
            proj_dim=128,        # 投影维度
            tau=0.07,            # 温度参数
            use_bn=True,
            use_dhn=False        # 禁用DHN (Plan B不需要)
        ),
        lambda1=0.005,           # 极轻量级 (原来0.05的1/10)
        lambda2=0.0,             # 禁用DHN
        lambda3=0.0,             # 禁用Domain Alignment
        use_domain_alignment=False,  # 修正: 不是domain_confusion
        use_dhn=False,           # 修正: 小写dhn
        use_msp=False,           # 禁用MSP
        
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # KAIST: person only
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
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
            nms_pre=2000,      # 增加proposals
            max_per_img=1500,  # 增加保留数量
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
            score_thr=0.35     # 降低阈值,增加recall
        ),
        rcnn=dict(
            score_thr=0.4,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

# ============================================================================
# Dataset Configuration - KAIST
# ============================================================================
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

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
    batch_size=2,  # 临时降低以减少内存压力和初始化开销
    num_workers=0,  # Windows compatibility
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        data_prefix=dict(sub_data_root='./'),
        # 仅单类 person，与 bbox_head.num_classes=1 保持一致（与 KAISTDataset METAINFO 一致）
        metainfo=dict(classes=('person',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

# ============================================================================
# Optimizer Configuration - Ultra-Low Learning Rate
# ============================================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,           # 超低学习率 (原来4e-5的1/4)
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.05)  # backbone极保守 (5e-7)
        }
    ),
    clip_grad=dict(max_norm=3.0, norm_type=2)
)

# ============================================================================
# Learning Rate Schedule - Progressive MACL Warmup
# ============================================================================
param_scheduler = [
    # Stage 1: Warmup (前1000 iters)
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Stage 2: Constant LR with Progressive Lambda
    dict(
        type='ConstantLR',
        factor=1.0,
        by_epoch=True,
        begin=0,
        end=6
    )
]

# Note: Lambda1渐进式增长需要在训练代码中手动实现
# Epoch 1-2: lambda1=0.005
# Epoch 3-4: lambda1=0.0075 (1.5x)
# Epoch 5-6: lambda1=0.01125 (1.5x again)

# ============================================================================
# Training Configuration
# ============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=6,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Runtime Configuration
# ============================================================================
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='pascal_voc/mAP',
        rule='greater',
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 暂时禁用可视化以避免初始化卡住
    # visualization=dict(type='DetVisualizationHook')
)

# 注册Progressive Lambda Hook
custom_imports = dict(
    imports=['configs.llvip.progressive_lambda_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='ProgressiveLambdaHook',
        milestones=[3, 5],  # Epoch 3和5时增加lambda1
        gamma=1.5,          # 每次增加1.5倍
        initial_lambda=0.005  # 初始值
    )
]

env_cfg = dict(
    cudnn_benchmark=False,
    # Windows 不支持 fork，多数情况下会导致无输出假死，改为 spawn
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# ============================================================================
# Resume Configuration - 从最佳Epoch 2恢复
# ============================================================================
load_from = 'work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth'
resume = False

# ============================================================================
# Work Directory
# ============================================================================
work_dir = './work_dirs/stage2_1_planB_macl_rescue'

# ============================================================================
# Auto Scale Learning Rate (disabled for manual control)
# ============================================================================
auto_scale_lr = dict(enable=False, base_batch_size=16)
