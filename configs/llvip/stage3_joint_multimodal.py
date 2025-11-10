"""
Stage3: Joint Multimodal Training
Target: Joint training with KAIST and M3FD datasets
Person-only multimodal alignment experiment (LLVIP -> KAIST -> M3FD)
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

# ========== Data Processing Pipeline ==========
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction', 'modality')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'modality')
    )
]

# ========== Model Configuration ==========
model = dict(
    type='FasterRCNN',
    neck=dict(
        type='FPN',
        use_msp=True,
        msp_module=dict(
            type='MSPReweight',
            channels=256,
            reduction=16
        )
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=1),
        use_macl=True,
        macl_head=dict(
            type='MACLHead',
            in_dim=256,
            proj_dim=128,
            temperature=0.07,
            use_dhn=True,
            dhn_cfg=dict(K=8192, m=0.99)
        ),
        use_msp=True,
        use_dhn=True,
        lambda1=1.0,
        lambda2=0.5,
        lambda3=0.1,
    )
)

# ========== Multi-Dataset Training Configuration ==========
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='KAISTDataset',
                data_root='C:/KAIST_processed/',
                ann_file='C:/KAIST_processed/ImageSets/train.txt',
                data_prefix=dict(sub_data_root='C:/KAIST_processed'),
                ann_subdir='Annotations',
                return_modality_pair=False,
                pipeline=train_pipeline,
            ),
            dict(
                type='M3FDDataset',
                data_root='C:/M3FD_processed/',
                ann_file='C:/M3FD_processed/ImageSets/train.txt',
                data_prefix=dict(sub_data_root='C:/M3FD_processed'),
                ann_subdir='Annotations',
                return_modality_pair=False,
                pipeline=train_pipeline,
            ),
        ]
    )
)

# ========== Validation Configuration (M3FD) ==========
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='M3FDDataset',
        data_root='C:/M3FD_processed/',
        ann_file='C:/M3FD_processed/ImageSets/val.txt',
        data_prefix=dict(sub_data_root='C:/M3FD_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
        test_mode=True,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

# ========== Evaluation Configuration ==========
val_evaluator = dict(
    type='CocoMetric',
    ann_file='C:/M3FD_processed/annotations_coco_format.json',
    metric='bbox',
    format_only=False,
)
test_evaluator = val_evaluator

# ========== Learning Rate Scheduler ==========
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
        T_max=12,
        eta_min=1e-6,
        begin=0,
        end=12,
        by_epoch=True
    )
]

# ========== Optimizer Configuration ==========
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# ========== Training Configuration ==========
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ========== Load Stage2 Weights ==========
load_from = 'work_dirs/stage2_kaist_domain_ft/epoch_12.pth'

# ========== Checkpoint Hook ==========
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3
    )
)

# ========== Output Directory ==========
work_dir = './work_dirs/stage3_joint_multimodal'

# ========== 冻结监控 Hook（将在 freezehook 变体中附加） ==========
custom_hooks = []
