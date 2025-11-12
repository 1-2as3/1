"""Stage2.1 KAIST Detection-Only Fine-Tune (Curriculum Phase 1)

Focus: Stabilize detection head on KAIST without strong contrastive/domain signals.
"""
from mmengine.config import read_base

# Ensure custom modules are importable
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.engine.hooks.tsne_visual_hook',
        'mmdet.engine.hooks.domain_weight_warmup_hook',
        'mmdet.engine.hooks.early_stop_hook',
        'mmdet.engine.hooks.lambda_warmup_hook',
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

# Model modifications
"""NOTE: We import the base Faster R-CNN definition above and mutate it
in-place instead of reassigning `model`, to preserve backbone/rpn/train/test cfg.
Also switch to the paired data preprocessor for thermal/RGB pairs."""

# data preprocessor for paired modalities
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# Keep existing FPN settings; just ensure MSP disabled in phase 1
model['neck'].update(dict(use_msp=False))

# Extend roi_head with curriculum flags & MACL head; retain base extractor/head
model['roi_head']['bbox_head']['num_classes'] = 1
model['roi_head'].update(dict(
    use_macl=False,     # ✅ DISABLED: Pure detection stabilization
    use_msp=False,
    use_dhn=False,
    use_domain_alignment=False,
    lambda1=0.0,        # ✅ Zero contrastive loss
    lambda2=0.0,
    lambda3=0.0,
))
# Remove MACL head since use_macl=False
# model['roi_head']['macl_head'] = dict(
#     type='MACLHead',
#     in_dim=256,
#     proj_dim=128,
#     temperature=0.07,
#     use_dhn=False,
# )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.005),  # ✅ Reduced from 1e-4
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    ),
    clip_grad=dict(max_norm=3.0, norm_type=2)  # ✅ Gradient clipping for stability
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=200, by_epoch=False, start_factor=0.001),
    dict(type='CosineAnnealingLR', begin=1, end=5, by_epoch=True, eta_min=1e-6),  # ✅ 5 epochs
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)  # ✅ Short recovery run

# Dataset (KAIST VOC)
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
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
        return_modality_pair=True,
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
        data_prefix=dict(sub_data_root=data_root),
        ann_file='ImageSets/val.txt',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
        ],
        return_modality_pair=True,
    )
)

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

custom_hooks = [
    dict(
        type='EarlyStopHook', 
        metric='pascal_voc/mAP', 
        threshold=0.60,  # ✅ Raised from 0.55 to react faster
        patience=2,      # ✅ Reduced from 3 to 2
        begin=1,         # ✅ Start monitoring from epoch 1
        verbose=True
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        max_keep_ckpts=3,  # ✅ Keep last 3 checkpoints
        save_best='pascal_voc/mAP', 
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=20)  # ✅ More frequent logging
)

work_dir = './work_dirs/stage2_1_kaist_detonly'

# Disable test loop for preflight/runner construction simplicity
test_dataloader = None
test_evaluator = None
test_cfg = None
