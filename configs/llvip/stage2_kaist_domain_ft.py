"""
阶段二：KAIST 域自适应微调
目标：在 KAIST 数据集上进行域自适应，冻结 backbone，启用所有模块
Person-only multimodal alignment experiment (LLVIP → KAIST)
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

# ========== 模型配置 ==========
# 说明：
# 1. 通过 read_base() 继承 faster_rcnn_r50_fpn.py 的完整配置
# 2. 这里的 model dict 会与基础配置深度合并（仅在训练流程中生效）
# 3. 测试脚本中需要手动加载基础配置并合并（参见 test_complete.py）
# 4. 自定义标志（use_macl 等）在 FPN 和 StandardRoIHead 中已实现
model = dict(
    type='FasterRCNN',
    # 启用 Neck (FPN) 的 MSP 模块
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
        bbox_head=dict(num_classes=1),  # Person-only detection (unified with LLVIP)
        # 启用自定义模块
        use_macl=True,           # ✅ 启用 MACL (Cross-Modal Alignment Loss)
        macl_head=dict(
            type='MACLHead',
            in_dim=256,
            proj_dim=128,
            temperature=0.07,
            use_dhn=True,
            dhn_cfg=dict(K=8192, m=0.99)
        ),
        use_msp=True,            # ✅ 启用 MSP (Modality-Specific Pooling)
        use_dhn=True,            # ✅ 启用 DHN (Dynamic Harmonization Network)
        use_domain_loss=True,    # ✅ 启用域对齐损失
    )
)

# ========== 加载阶段一模型 ==========
# 注意：请先运行 Stage1 训练生成此文件，或使用官方预训练权重
# 官方 Faster R-CNN ResNet50-FPN 权重（临时方案）：
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = './work_dirs/stage1_llvip_pretrain/epoch_latest.pth'

# ========== 优化器配置（冻结 backbone，降低学习率）==========
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0003),  # 降低学习率至 3e-4 for fine-tuning
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0)  # 设置学习率为0
        }
    )
)

# ========== 自定义 Hook：真正冻结 Backbone ==========
# 重要：lr_mult=0 只是设置学习率为0，参数仍然参与计算
# 需要显式设置 requires_grad=False 来真正冻结参数
custom_hooks = [
    dict(
        type='RuntimeInfoHook',  # 使用内置hook类型作为基类
        priority='VERY_LOW'
    )
]

# 注意：真正的冻结需要在训练脚本中手动实现
# 在 tools/train.py 中添加（或创建自定义 hook）：
#   for name, param in model.named_parameters():
#       if 'backbone' in name:
#           param.requires_grad = False

# ========== 学习率调度器（使用 CosineAnnealing）==========
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=12,
    eta_min=1e-6,
    by_epoch=True
)

# ========== 运行配置 ==========
work_dir = './work_dirs/person_only_stage2'

# ========== 数据集（KAIST processed VOC） ==========
data = dict(
    # 说明：KAISTDataset 的 return_modality_pair=True 会返回原始可见光+红外图像对，
    # 不走标准 pipeline，仅适合自定义多模态训练循环。为保证与 MMDet 标准训练兼容，默认关闭配对模式。
    train=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/train.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        # return_modality_pair=True,  # 如需双模态配对训练，请开启，并使用自定义训练循环
        return_modality_pair=False,
    ),
    val=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/val.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
    ),
    test=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/test.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        ann_subdir='Annotations',
        return_modality_pair=False,
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
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/test.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
        # img_subdir 留空，由 KAISTDataset.parse_data_info 动态选择 visible/infrared
        ann_subdir='Annotations',
        return_modality_pair=False,  # 标准测试流程使用单模态（由样本ID决定 visible / infrared）
        pipeline=test_pipeline,
    )
)

# ========== 评估配置 ==========
test_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_cfg = dict(type='TestLoop')

# 禁用验证（默认不启用 val，可手动开启）
val_evaluator = None
val_cfg = None
val_dataloader = None
