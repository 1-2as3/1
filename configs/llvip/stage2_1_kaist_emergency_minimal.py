"""Stage2.1-C: Emergency Minimal Config (Copy Stage1 structure exactly)

Hypothesis: Something in Plan A config is incompatible. 
Copy Stage1's exact structure and only change to KAIST dataset.
"""
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmdet.datasets.kaist_dataset'],
    allow_failed_imports=False
)

# Load Stage1 checkpoint  
load_from = 'work_dirs/stage1_longrun_full/epoch_21.pth'

# Model settings - minimal changes
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
    )
)

# Dataset
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

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
        pipeline=train_pipeline,
        return_modality_pair=False
    )
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

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
        pipeline=val_pipeline,
        return_modality_pair=False
    )
)

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,  # Delete inherited optimizer config completely
        type='AdamW', 
        lr=5e-5, 
        betas=(0.9, 0.999),
        weight_decay=5e-4
    )
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, by_epoch=False, start_factor=0.001),
    dict(type='CosineAnnealingLR', begin=1, end=5, by_epoch=True, eta_min=1e-6),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='pascal_voc/mAP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

work_dir = './work_dirs/stage2_1_emergency'
