"""
阶段一：LLVIP 预训练
目标：在 LLVIP 数据集上训练基础模型，启用 MACL 和 MSP
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

# 清理命名空间，避免将函数对象写入配置（防止 pickle 报错）
del read_base

# ========== 模型配置（在基础模型上更新，而非覆盖） ==========
# 基础模型已通过 read_base 导入到变量 `model` 中，这里仅更新所需字段

# 保持BatchNorm（GroupNorm初始化开销过大）
# 但设置 frozen_stages=1 冻结早期层减少梯度计算
model.setdefault('backbone', {}).update(dict(
    frozen_stages=1,  # 冻结stage1减少计算
    norm_eval=True    # eval模式的BN更稳定
))

model.setdefault('roi_head', {}).update(dict(
    type='StandardRoIHead',
    use_macl=True,   # ✅ 启用 MACL
    use_msp=True,    # ✅ 启用 MSP                         多余，见35行
    use_dhn=False,   # ❌ 阶段一不使用 DHN
    bbox_head=dict(num_classes=1)  # LLVIP 只有 person 类别
))

# 在 FPN 颈部启用 MSP 模块
model.setdefault('neck', {}).update(dict(
    use_msp=True,
    msp_module=dict(type='MSPReweight', channels=256)
))

# ========== 数据增强管道 ==========
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# ========== 数据集配置 ==========
train_dataloader = dict(
    batch_size=2,  # 降低batch_size减少内存和计算压力
    num_workers=0,  # Windows调试模式：先设为0避免spawn问题
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='LLVIPDataset',
        data_root='C:/LLVIP/LLVIP/',
        ann_file='ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root=''),
        img_subdir='visible/train',  # ✅ LLVIP训练集图像在此目录
        ann_subdir='Annotations',  # ✅ 指定标注目录
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        return_modality_pair=True  # ✅ 启用配对模态输入（可见光+红外）
    )
)

# ========== 测试管道 ==========
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'modality')
    )
]

# ========== 测试数据加载器 ==========
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
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

# ========== 评估配置 ==========
test_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_cfg = dict(type='TestLoop')

# 禁用验证（快速测试模式）
val_evaluator = None
val_cfg = None
val_dataloader = None

# ========== 默认 Hooks 配置：科研监控扩展 ==========
default_hooks = dict(
    metrics_export=dict(
        type='MetricsExportHook',
        interval=1,
        out_dir='work_dirs/metrics_logs',  # 自动添加 run_id 子目录
        clip_grad_max_norm=5.0,
        record_grad_norm=True,
        enable_html_report=True,
        enable_multi_exp_compare=True,
        sync_best_ckpt_to_workdir=True  # 同步最佳 checkpoint 到 work_dir
    ),
    tsne_visual=dict(
        type='TSNEVisualHook',
        interval=1,
        num_samples=200,
        out_dir='work_dirs/tsne_vis',
        perplexity=30
    )
)

# 可选：额外自定义 Hooks（保留清显存等轻量 Hook）
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True)
]

# ========== 优化器配置 ==========
# 使用SGD替代AdamW（更稳定，内存占用更小）+ 梯度裁剪
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# 禁用混合精度训练（避免数值不稳定）
fp16 = None

# ========== 环境配置（Windows 友好） ==========
# 避免 Windows 下多进程使用 fork 导致问题，强制使用 spawn
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
)

# ========== 运行配置 ==========
work_dir = './work_dirs/stage1_llvip_pretrain'

# 覆盖训练循环配置（确保命令行参数生效）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
