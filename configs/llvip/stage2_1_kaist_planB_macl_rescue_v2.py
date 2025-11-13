"""
Plan B: MACL Progressive Rescue Configuration (Base Config Version)
===================================================================
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

from mmengine.config import read_base

# 注册 progressive lambda hook
custom_imports = dict(
    imports=['configs.llvip.progressive_lambda_hook'],
    allow_failed_imports=False
)

# 使用 base configs (更稳定的方式)
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

# ============================================================================
# Load Checkpoint
# ============================================================================
load_from = 'work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth'
resume = False

# ============================================================================
# Model Configuration - MACL Rescue Mode
# ============================================================================
# 修改检测头为单类
model['roi_head']['bbox_head']['num_classes'] = 1

# 添加 MACL 相关配置
model['roi_head'].update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_msp=False,
    use_dhn=False,
    use_domain_alignment=False,
    lambda1=0.005,  # 极轻量级 MACL
    lambda2=0.0,
    lambda3=0.0,
    macl_head=dict(
        type='MACLHead',
        in_dim=1024,
        proj_dim=128,
        tau=0.07,
        use_bn=True,
        use_dhn=False
    )
))

# 调整 test_cfg 提高 recall
model['test_cfg']['rpn']['nms_pre'] = 2000
model['test_cfg']['rpn']['max_per_img'] = 1500
model['test_cfg']['rpn']['score_thr'] = 0.35

# ============================================================================
# Optimizer Configuration - Ultra-Low Learning Rate
# ============================================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,           # 超低学习率
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
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='ConstantLR',
        factor=1.0,
        by_epoch=True,
        begin=0,
        end=6
    )
]

# ============================================================================
# Training Configuration
# ============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=6,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = None  # 明确禁用 test

# ============================================================================
# Dataset Configuration - KAIST
# ============================================================================
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,  # Windows 上用少量 worker 反而更稳定
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        # Plan B 使用单模态训练（仅 lwir），不需要配对
        # return_modality_pair=False  # 默认值，明确注释
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = None  # 明确禁用
test_evaluator = None   # 明确禁用

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# ============================================================================
# Hooks Configuration
# ============================================================================
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
    sampler_seed=dict(type='DistSamplerSeedHook')
    # 禁用可视化避免卡住
)

# Progressive Lambda Hook
custom_hooks = [
    dict(
        type='ProgressiveLambdaHook',
        milestones=[3, 5],
        gamma=1.5,
        initial_lambda=0.005
    )
]

# ============================================================================
# Runtime Configuration
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

work_dir = './work_dirs/stage2_1_planB_macl_rescue'
