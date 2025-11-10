"""
阶段二：KAIST 域自适应微调（移除域对齐项以确保可构建）
目标：在 KAIST 数据集上进行微调，冻结 backbone，启用 MACL/MSP/DHN
本文件基于 `stage2_kaist_domain_ft.py`，仅移除 roi_head.use_domain_loss 以避免 BaseRoIHead 参数不兼容。
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa: F401,F403
    from .._base_.schedules.schedule_1x import *       # noqa: F401,F403
    from .._base_.default_runtime import *             # noqa: F401,F403

# ========== 模型配置（仅移除 use_domain_loss） ==========
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
        type='AlignedRoIHead',
        bbox_head=dict(num_classes=1),  # Person-only detection
        # 自定义模块
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
        use_domain_aligner=True,
        domain_aligner=dict(
            type='DomainAligner',
            level='fpn_p3',
            method='MMD',
            loss_weight=0.1,
            normalize=True,
            mmd_kernels=(1.0, 2.0, 4.0)
        ),
        lambda_domain=0.1,
    )
)

# ========== 加载阶段一模型 ==========
load_from = 'work_dirs/stage1_longrun_full/epoch_21.pth'

# ========== 优化器配置（冻结 backbone，降低学习率）==========
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0003),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0)
        }
    )
)

# ========== Hook 与调度器 ==========
custom_hooks = [
    dict(type='RuntimeInfoHook', priority='VERY_LOW')
]

param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=12,
    eta_min=1e-6,
    by_epoch=True
)

work_dir = './work_dirs/person_only_stage2'

# ========== 数据集（KAIST processed VOC） ==========
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'modality'))
]

data = dict(
    train=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/train.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=True,
        pipeline=train_pipeline,
    ),
    val=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/val.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
        pipeline=test_pipeline,
    ),
    test=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/test.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
        pipeline=test_pipeline,
    )
)



test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/test.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
        pipeline=test_pipeline,
    )
)

test_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_cfg = dict(type='TestLoop')

val_evaluator = None
val_cfg = None
val_dataloader = None
