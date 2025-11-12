"""Stage2.1-A: Pure Detection Recovery (Single Modality Test)

EMERGENCY DEBUGGING VERSION:
- Disable return_modality_pair to test if paired data loading is the issue
- Everything else same as Plan A configuration
"""
from mmengine.config import read_base

custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.engine.hooks.early_stop_hook',
        'mmdet.datasets.kaist_dataset',
    ],
    allow_failed_imports=False,
)

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

# Load Stage1 checkpoint
load_from = 'work_dirs/stage1_longrun_full/epoch_21.pth'

# Model: Pure detection mode
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

model['neck'].update(dict(use_msp=False))

model['roi_head']['bbox_head']['num_classes'] = 1
model['roi_head'].update(dict(
    use_macl=False,
    use_msp=False,
    use_dhn=False,
    use_domain_alignment=False,
    lambda1=0.0,
    lambda2=0.0,
    lambda3=0.0,
))

# Conservative optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=5e-4),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    ),
    clip_grad=dict(max_norm=3.0, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, by_epoch=False, start_factor=0.001),
    dict(type='CosineAnnealingLR', begin=1, end=5, by_epoch=True, eta_min=1e-6),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

# Dataset - SINGLE MODALITY MODE (no paired loading)
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_dataloader = dict(
    batch_size=4,
    num_workers=0,  # Windows fix
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(sub_data_root=data_root),
        ann_file='ImageSets/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        return_modality_pair=False,  # 🔴 DISABLED for debugging
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
        data_prefix=dict(sub_data_root=data_root),
        ann_file='ImageSets/val.txt',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
        ],
        return_modality_pair=False,  # 🔴 DISABLED for debugging
    )
)

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# Stricter early stopping
custom_hooks = [
    dict(type='EarlyStopHook', 
         metric='pascal_voc/mAP', 
         threshold=0.58,
         patience=2,
         begin=2, 
         verbose=True)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='pascal_voc/mAP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

work_dir = './work_dirs/stage2_1_pure_detection_single'

# FP16 for memory efficiency
fp16 = dict(loss_scale='dynamic')

# Disable test
test_dataloader = None
test_evaluator = None
test_cfg = None
