"""
Stage1 长程高质量预训练 (方案B)
- 24 epochs
- AdamW + CosineAnnealingLR
- 解冻 backbone（frozen_stages=0）
- 可配置多尺度 + 颜色扰动 + 随机裁剪
- MACLHead 切换为 GN + Dropout
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa: F401,F403
    from .._base_.schedules.schedule_1x import *       # 将被覆盖关键项
    from .._base_.default_runtime import *             # 基础 hooks

del read_base

# --- 模型调整 ---
model.setdefault('backbone', {}).update(dict(
    frozen_stages=0,  # 解冻全部层
    norm_eval=False,  # 允许 BN 更新，提高适应性
))

# RoIHead 中启用 MACL + MSP（保持原语义）
model.setdefault('roi_head', {}).update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_msp=True,
    use_dhn=True,
    macl_head=dict(
        type='MACLHead',
        in_dim=256,
        proj_dim=128,
        norm_type='GN',
        norm_groups=32,
        hidden_dims=(256, 128),
        dropout=0.1,
        # 长跑集成 DHN：初期启用但配合调度 Hook 降低难度
        use_dhn=True,
        dhn_cfg=dict(queue_size=8192, momentum=0.995, top_k=64),
    ),
    bbox_head=dict(num_classes=1)
))

# FPN 启用 MSP（如果基础模型未配置）
model.setdefault('neck', {}).update(dict(
    use_msp=True,
    msp_module=dict(type='MSPReweight', channels=256)
))

# --- 数据增强 ---
_base_resize = dict(type='Resize', scale=(640, 640), keep_ratio=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoice', transforms=[
        [dict(type='Resize', scale=s, keep_ratio=True) for s in [(640,640),(672,672),(704,704)]],
        [dict(type='Resize', scale=(640,640), keep_ratio=True), dict(type='RandomCrop', crop_size=(560,560), allow_negative_crop=True)]
    ]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,  # 提升 BN 统计稳定性
    num_workers=0,  # Windows 可适当增至 2~4（测试后再改）
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    _base_resize,
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
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

# --- Hooks 扩展 ---
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    # 质量对齐可视化与指标
    dict(type='TSNEVisualHook', interval=1, num_samples=200),
    # DHN 分段调度：0~2:64, 3~7:96, 8~15:128, >=16:128; momentum 下调到 0.99
    dict(type='DHNScheduleHook', milestones=[3, 8, 16],
         topk_stages=[64, 96, 128, 128], momentum_stages=[0.995, 0.99, 0.99, 0.99]),
    # EMA 稳定训练
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)
]

# --- 优化器与调度 ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=5e-4),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=800),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=24, T_max=24, eta_min=1e-6)
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

test_cfg = dict(type='TestLoop')

# --- 环境 ---
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0)
)

work_dir = './work_dirs/stage1_longrun'

fp16 = None  # 可选：若稳定后加 amp=True
