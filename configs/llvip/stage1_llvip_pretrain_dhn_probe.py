"""
Stage1 DHN 探测配置 (2-3 epoch 快速评估)
- 目标：验证开启 DHN (Dynamic Hard Negative) 后 MACL 收敛与对齐稳定性影响
- 训练轮数：3
- 保持核心结构，缩短调度周期 (Cosine T_max=3)
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.schedules.schedule_1x import *       # 将覆盖调度与优化器
    from .._base_.default_runtime import *

del read_base

# --- 模型：解冻 + 启用 MACL + DHN ---
model.setdefault('backbone', {}).update(dict(
    frozen_stages=0,
    norm_eval=False,
))

model.setdefault('roi_head', {}).update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_msp=True,
    use_dhn=True,  # 显式开启 DHN
    macl_head=dict(
        type='MACLHead',
        in_dim=256,
        proj_dim=128,
        norm_type='GN',
        norm_groups=32,
        hidden_dims=(256,128),
        dropout=0.1,
        use_dhn=True,  # 传递给 MACLHead 以启用困难负样本
        dhn_cfg=dict(queue_size=4096, momentum=0.995, top_k=128),
    ),
    bbox_head=dict(num_classes=1)
))

model.setdefault('neck', {}).update(dict(
    use_msp=True,
    msp_module=dict(type='MSPReweight', channels=256)
))

# --- 数据管道（不做复杂增强，保证可复现性）---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
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
        type='LLVIPDataset',
        data_root='C:/LLVIP/LLVIP/',
        ann_file='ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root=''),
        img_subdir='visible/train',
        ann_subdir='Annotations',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        return_modality_pair=True
    )
)

# 测试保持最简
_base_resize = dict(type='Resize', scale=(640,640), keep_ratio=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    _base_resize,
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LLVIPDataset',
        data_root='C:/LLVIP/LLVIP/',
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root=''),
        img_subdir='visible/train',
        ann_subdir='Annotations',
        pipeline=test_pipeline,
        return_modality_pair=True
    )
)

val_cfg = None
val_dataloader = None
val_evaluator = None

test_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# --- Hooks ---
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    # 每个 epoch 采样少量嵌入进行 t-SNE 可视化，并记录对齐质量指标
    dict(type='TSNEVisualHook', interval=1, num_samples=200)
]

# --- 优化与调度（短程 Cosine） ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=5e-4),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=5.0)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=3, T_max=3, eta_min=1e-6)
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)

test_cfg = dict(type='TestLoop')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0)
)

work_dir = './work_dirs/stage1_dhn_probe'

fp16 = None
