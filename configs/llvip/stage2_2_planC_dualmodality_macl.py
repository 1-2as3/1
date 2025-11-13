"""
Plan C: Dual-Modality MACL Rescue Configuration
================================================
目标: 激活真正的MACL双模态对比学习,重建跨域语义一致性
容忍区间: 0.52 ≤ mAP < 0.60 (低于0.52=梯度错向,0.55左右可通过调参救回)

核心策略:
1. 双模态配对数据 (visible + infrared) - MACL的前提条件
2. 保守MACL权重 (lambda1=0.01, 初始阶段1/5原始值) - 避免主导训练
3. 禁用MSP/DHN - 专注MACL收敛
4. 温和学习率 (lr=5e-5, backbone=5e-6) - 防止特征空间震荡
5. 快速验证 (3 epoch) - 若mAP<0.52立即中断

关键监控指标:
- loss_macl 必须出现且收敛 (从0.5降至0.2左右)
- loss_det (cls+bbox) 稳定下降
- mAP在epoch 1-2应回升至0.55+
- grad_norm < 10 表示稳定收敛

失败判定:
- Epoch 1 mAP < 0.52 → 立即停止 (特征崩溃)
- Epoch 2 mAP < 0.55 且loss_macl不收敛 → 调整lambda1/lr
- Epoch 3 mAP < 0.58 → 考虑切换策略
"""

from mmengine.config import read_base

# 注册自定义模块
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.datasets.kaist_dataset',
        'mmdet.engine.hooks.early_stop_hook',
    ],
    allow_failed_imports=False
)

# 使用 base configs
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

# ============================================================================
# Load Checkpoint - 从Stage 2.1 Epoch 2恢复
# ============================================================================
load_from = 'work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth'
resume = False

# ============================================================================
# Model Configuration - Dual-Modality MACL
# ============================================================================

# 关键:使用双模态数据预处理器
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

# 修改为单类检测
model['roi_head']['bbox_head']['num_classes'] = 1

# 添加 MACL 配置 (保守权重)
model['roi_head'].update(dict(
    type='StandardRoIHead',
    use_macl=True,        # 启用MACL
    use_msp=False,        # 先禁用MSP,专注MACL稳定
    use_dhn=False,        # 禁用DHN
    use_domain_alignment=False,  # 初期禁用
    lambda1=0.01,         # 保守MACL权重 (1/5原始值)
    lambda2=0.0,          # MSP权重
    lambda3=0.0,          # 域对齐权重
    macl_head=dict(
        type='MACLHead',
        in_dim=1024,      # RoI Head输出维度
        proj_dim=128,     # 投影维度
        tau=0.07,         # 温度参数
        use_bn=True,      # 使用BatchNorm稳定训练
        use_dhn=False
    )
))

# 调整检测配置提高召回
model['test_cfg']['rpn']['nms_pre'] = 2000
model['test_cfg']['rpn']['max_per_img'] = 1500
model['test_cfg']['rpn']['score_thr'] = 0.35

# ============================================================================
# Optimizer Configuration - 温和学习率
# ============================================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,           # 温和学习率 (比Plan B的1e-5高5倍)
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)  # backbone=5e-6
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2)  # 放宽梯度裁剪
)

# ============================================================================
# Learning Rate Schedule - Warmup + Constant
# ============================================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500         # 500 iter warmup
    ),
    dict(
        type='ConstantLR',
        factor=1.0,
        by_epoch=True,
        begin=0,
        end=6           # 6 epoch constant LR
    )
]

# ============================================================================
# Training Configuration
# ============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=6,       # 快速验证周期
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = None

# ============================================================================
# Dataset Configuration - KAIST Paired Mode
# ============================================================================
dataset_type = 'KAISTDataset'
data_root = 'C:/KAIST_processed/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=2,       # 降低batch size避免死锁 (Windows双模态加载问题)
    num_workers=0,      # Windows多进程+双模态=死锁,改为0
    persistent_workers=False,  # 禁用persistent避免资源竞争
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        return_modality_pair=True,   # 关键:启用双模态配对!
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,      # 同样改为0避免死锁
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(sub_data_root='./'),
        metainfo=dict(classes=('person',)),
        test_mode=True,
        pipeline=test_pipeline,
        return_modality_pair=True,   # 验证时也用双模态
    )
)

test_dataloader = None
test_evaluator = None

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# ============================================================================
# Hooks Configuration
# ============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='pascal_voc/mAP',
        rule='greater',
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Early Stop Hook - 快速失败机制
custom_hooks = [
    dict(
        type='EarlyStopHook',
        metric='pascal_voc/mAP',
        threshold=0.52,      # 低于0.52立即停止
        patience=2,          # 2个epoch不提升则停止
        begin=1,             # 从epoch 1开始监控
        verbose=True
    )
]

# TensorBoard可视化配置
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 启用TensorBoard
    ],
    name='visualizer'
)

# ============================================================================
# Runtime Configuration
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

work_dir = './work_dirs/stage2_2_planC_dualmodality_macl'

# ============================================================================
# 调试提示
# ============================================================================
"""
启动训练前必须检查:
1. train.txt 中是否包含visible和infrared配对样本
   例: set00_V000_visible_I01216 和 set00_V000_lwir_I01216
   
2. 运行命令:
   python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py

3. 关键监控指标 (前50个iter):
   - loss_macl 是否出现 (如果没有=配对失败)
   - loss_total ≈ 0.2-0.3 (检测loss) + 0.1-0.5 (MACL loss)
   - grad_norm 应在 5-15 之间
   
4. Epoch 1结束后判断:
   - mAP ≥ 0.55 → 继续训练
   - 0.52 ≤ mAP < 0.55 → 调整lambda1=0.005或lr=3e-5
   - mAP < 0.52 → 立即停止,检查数据配对

5. 若Epoch 1成功,可在Epoch 3后逐步启用:
   - lambda2=0.005 (MSP)
   - lambda3=0.005 (Domain Alignment)
"""
