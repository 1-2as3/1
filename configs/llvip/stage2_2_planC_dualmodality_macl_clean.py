"""
Plan C: Dual-Modality MACL Rescue Configuration
================================================
Target: Activate real MACL dual-modality contrastive learning
Tolerance: 0.52 <= mAP < 0.60

Core Strategy:
1. Dual-modality paired data (visible + infrared)
2. Conservative MACL weight (lambda1=0.01)
3. Disable MSP/DHN - Focus on MACL convergence
4. Moderate learning rate (lr=5e-5, backbone=5e-6)
5. Quick validation (3 epochs) - Stop if mAP<0.52

Key Monitoring Metrics:
- loss_macl must appear and converge (0.5 -> 0.2)
- loss_det (cls+bbox) stable decrease
- mAP should rise to 0.55+ at epoch 1-2
- grad_norm < 10 indicates stable convergence

Failure Criteria:
- Epoch 1 mAP < 0.52 -> Stop immediately
- Epoch 2 mAP < 0.55 and loss_macl not converging -> Adjust lambda1/lr
- Epoch 3 mAP < 0.58 -> Consider switching strategy
"""

from mmengine.config import read_base

# Register custom modules
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.datasets.kaist_dataset',
        'mmdet.engine.hooks',  # Import hooks package (includes EarlyStopHook via __init__)
    ],
    allow_failed_imports=False
)

# Use base configs
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
# Model Configuration - Dual-Modality MACL
# ============================================================================

# CRITICAL: Use paired data preprocessor
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# Single class detection
model['roi_head']['bbox_head']['num_classes'] = 1

# Add MACL configuration (conservative weight)
model['roi_head'].update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_msp=False,
    use_dhn=False,
    use_domain_alignment=False,
    lambda1=0.01,
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

# Adjust detection config for higher recall
model['test_cfg']['rpn']['nms_pre'] = 2000
model['test_cfg']['rpn']['max_per_img'] = 1500
model['test_cfg']['rpn']['score_thr'] = 0.35

# ============================================================================
# Optimizer Configuration
# ============================================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# ============================================================================
# Learning Rate Schedule
# ============================================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500
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
test_cfg = None

# ============================================================================
# Dataset Configuration - KAIST Paired Mode
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

# FIXED: num_workers=0 to avoid deadlock on Windows with dual-modality loading
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
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
        return_modality_pair=True,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        test_mode=True,
        pipeline=test_pipeline,
        return_modality_pair=True,
    )
)

test_dataloader = None
test_evaluator = None

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
)

# Early Stop Hook
custom_hooks = [
    dict(
        type='EarlyStopHook',
        metric='pascal_voc/mAP',
        threshold=0.52,
        patience=2,
        begin=1,
        verbose=True
    )
]

# TensorBoard visualization
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

# ============================================================================
# Runtime Configuration
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

work_dir = './work_dirs/stage2_2_planC_dualmodality_macl'
