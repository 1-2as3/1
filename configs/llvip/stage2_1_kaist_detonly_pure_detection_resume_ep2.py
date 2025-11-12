"""Stage2.1-A Resume from Best Ep2 (Constant LR, No EarlyStop)

Purpose:
- Resume from best checkpoint at epoch 2 (mAP=0.5884)
- Disable EarlyStop to avoid premature stop on short-term mAP dips
- Use constant LR for stability (no cosine decay)

Expected:
- Allow 3 more epochs to recover to >=0.63 if trend resumes
"""
from mmengine.config import read_base

custom_imports = dict(
    imports=[
        'mmdet.datasets.kaist_dataset',
    ],
    allow_failed_imports=False,
)

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

# Load from best epoch 2
load_from = 'work_dirs/stage2_1_pure_detection/best_pascal_voc_mAP_epoch_2.pth'

# Model tweaks
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

# Dataset
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
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
        return_modality_pair=False,
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
        return_modality_pair=False,
    )
)

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# Train for a few more epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = None

# Optimizer: Constant LR for stability (no cosine)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=4e-5,  # slightly higher floor than epoch 4's 2.55e-5
        betas=(0.9, 0.999),
        weight_decay=5e-4,
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=3.0, norm_type=2),
)

# Keep LR constant
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, by_epoch=True, begin=0, end=1000),
]

# Disable EarlyStop
# custom_hooks = []

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='pascal_voc/mAP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

work_dir = './work_dirs/stage2_1_pure_detection'
