# ============================================================================
# Plan C for Linux: Full Cross-Modal Detection with MACL
# Target: mAP >= 0.60 through visible-infrared contrastive learning
# ============================================================================

_base_ = '../rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ============== Custom Module Registration ==============
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.datasets.kaist_dataset',
        'mmdet.engine.hooks'
    ],
    allow_failed_imports=False
)

# ============== Dataset Configuration ==============
dataset_type = 'KAISTDataset'
data_root = '/path/to/kaist_dataset/'  # !! MODIFY THIS !!

# Training pipeline (with augmentation)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='PackDetInputs')
]

# Validation pipeline (no augmentation)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# ============== DataLoader Configuration ==============
train_dataloader = dict(
    batch_size=4,  # Increase if GPU memory allows (8 or 16)
    num_workers=6,  # Linux can use fork multiprocessing
    persistent_workers=True,  # Cache workers for faster training
    pin_memory=True,  # Enable for faster GPU transfer
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.txt',  # !! VERIFY PATH !!
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        return_modality_pair=True  # KEY: Enable dual-modality loading
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.txt',  # !! VERIFY PATH !!
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_modality_pair=True  # KEY: Enable dual-modality loading
    )
)

test_dataloader = val_dataloader

# ============== Model Configuration ==============
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='PairedDetDataPreprocessor',  # Custom dual-modality preprocessor
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=3,  # person, people, cyclist
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')
    ),
    roi_head=dict(
        type='DynamicRoIHeadWithMACL',
        use_macl=True,  # Enable MACL contrastive learning
        lambda1=0.01,  # Conservative weight for stability
        lambda2=0.0,  # Disable domain alignment (single dataset)
        tau=0.07,  # Temperature for contrastive loss
        proj_dim=128,  # Projection dimension
        queue_size=1024,  # Memory queue size for hard negatives
        msp_enabled=True,  # Enable Modality Specific Pooling
        dhn_enabled=True  # Enable Dynamic Harmonization Network
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300
    )
)

# ============== Optimizer Configuration ==============
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # Ultra-low learning rate for fine-tuning
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            'projection_head': dict(lr_mult=5.0),  # Boost MACL head learning
            'msp_layers': dict(lr_mult=3.0),       # Boost MSP learning
            'queue': dict(decay_mult=0.0)          # Freeze memory queue
        }
    )
)

# ============== Training Schedule ==============
max_epochs = 30
stage2_num_epochs = 20
base_lr = 0.0001

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# ============== Runtime Configuration ==============
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Custom hooks
custom_hooks = [
    dict(
        type='EarlyStopHook',
        monitor='coco/bbox_mAP',
        patience=3,
        threshold=0.52,  # Stop if mAP drops below 0.52
        mode='max',
        begin=1
    )
]

# Environment configuration (Linux-optimized)
env_cfg = dict(
    cudnn_benchmark=True,  # Enable cuDNN benchmark for faster training
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # fork is faster on Linux
    dist_cfg=dict(backend='nccl')  # Use NCCL for multi-GPU training
)

# Visualization configuration
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logging configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Load pretrained weights
load_from = './work_dirs/stage1_rtmdet_kaist_final/epoch_48.pth'  # !! VERIFY PATH !!

# Resume configuration
resume = False
auto_scale_lr = dict(enable=False, base_batch_size=32)

# ============== Evaluation Configuration ==============
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.txt',  # !! VERIFY PATH !!
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator
