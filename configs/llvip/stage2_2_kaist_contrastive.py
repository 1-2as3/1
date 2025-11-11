"""Stage2.2 KAIST Contrastive + Domain Alignment (Curriculum Phase 2)

Start from best of Stage2.1, enable contrastive and domain losses with gentle warmups.
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

# Load best from Stage2.1
load_from = 'work_dirs/stage2_1_kaist_detonly/best_pascal_voc_mAP.pth'

fp16 = dict(loss_scale='dynamic')

"""Mutate base model in-place to keep backbone/rpn/train/test cfg intact.
Add MSP + MACL/DHN/domain alignment heads with warmup-enabled weights.
Switch to paired data preprocessor for dual-modality inputs."""

# paired data preprocessor
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# enable MSP on neck
model['neck'].update(dict(
    use_msp=True,
    msp_module=dict(type='MSPReweight', channels=256),
))

# roi head adjustments
model['roi_head']['bbox_head']['num_classes'] = 1
model['roi_head'].update(dict(
    use_macl=True,
    use_msp=True,
    use_dhn=True,
    use_domain_alignment=True,
    lambda1=0.1,   # -> 0.3 by warmup
    lambda2=0.05,  # -> 0.1 by warmup
    lambda3=0.05,  # domain alignment weight
))
model['roi_head']['macl_head'] = dict(
    type='MACLHead',
    in_dim=256,
    proj_dim=128,
    temperature=0.07,
    use_dhn=True,
    dhn_cfg=dict(queue_size=8192, momentum=0.99),
    enable_domain_loss=True,
    domain_weight=0.0,  # warm to 0.05 in 2 epochs
    mmd_kernels=(1.0, 2.0, 4.0),
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=8e-5, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=200, by_epoch=False, start_factor=0.001),
    dict(type='CosineAnnealingLR', begin=1, end=12, by_epoch=True, eta_min=8e-6),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

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

# Warmups: lambda1, lambda2 and domain_weight
custom_hooks = [
    # Early stop on poor mAP
    dict(type='EarlyStopHook', metric='pascal_voc/mAP', threshold=0.55, patience=3, begin=2, verbose=True),
    # Lambda warmups (0.1->0.3, 0.05->0.1)
    dict(
        type='LambdaWarmupHook',
        items=[
            dict(path='roi_head.lambda1', start=0.1, target=0.3),
            dict(path='roi_head.lambda2', start=0.05, target=0.1),
        ],
        warmup_epochs=6,
        verbose=True,
    ),
    # Domain weight warmup (0.0->0.05 over 2 epochs)
    dict(
        type='DomainWeightWarmupHook',
        attr_path='roi_head.macl_head.domain_weight',
        mode='linear', start=0.0, target=0.05, warmup_epochs=2, verbose=True
    ),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='pascal_voc/mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50)
)

work_dir = './work_dirs/stage2_2_kaist_contrastive'

# Disable test loop for preflight simplicity
test_dataloader = None
test_evaluator = None
test_cfg = None
