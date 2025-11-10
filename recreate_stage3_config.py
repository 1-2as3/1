"""重新创建 Stage3 配置文件"""
import os

config_content = '''"""
阶段三:多模态联合训练
目标:联合 KAIST 和 M3FD 数据集进行多模态训练
Person-only multimodal alignment experiment (LLVIP -> KAIST -> M3FD)
"""
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

# ========== 数据处理 Pipeline ==========
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

# ========== 模型配置 ==========
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
        bbox_head=dict(num_classes=1),  # Person-only detection (unified across all datasets)
        # 启用自定义模块
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
        # 损失权重配置 (在 StandardRoIHead 中需要实现相应的损失计算逻辑)
        lambda1=1.0,      # MACL (跨模态对齐) 权重
        lambda2=0.5,      # DHN (困难负样本) 权重
        lambda3=0.1,      # Domain Alignment 权重
    )
)

# ========== 多数据集训练配置 ==========
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # KAIST 数据集(红外+可见光)
            dict(
                type='KAISTDataset',
                data_root='C:/KAIST_processed/',
                ann_file='C:/KAIST_processed/ImageSets/train.txt',
                data_prefix=dict(sub_data_root='C:/KAIST_processed'),
                ann_subdir='Annotations',
                return_modality_pair=False,
                pipeline=train_pipeline,
            ),
            # M3FD 数据集(红外+可见光)
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

# ========== 验证集配置(M3FD) ==========
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

# ========== 评估配置 ==========
val_evaluator = dict(
    type='CocoMetric',
    ann_file='C:/M3FD_processed/annotations_coco_format.json',
    metric='bbox',
    format_only=False,
)
test_evaluator = val_evaluator

# ========== 学习率调度器 ==========
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

# ========== 优化器配置 ==========
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# ========== 训练配置 ==========
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ========== 加载 Stage2 权重 ==========
load_from = 'work_dirs/stage2_kaist_domain_ft/epoch_12.pth'

# ========== 训练记录 ==========
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3
    )
)

# ========== 元信息 ==========
work_dir = './work_dirs/stage3_joint_multimodal'
'''

# 删除旧文件并写入新文件
output_file = r'c:\Users\Xinyu\mmdetection\configs\llvip\stage3_joint_multimodal.py'
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"已删除旧文件: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(config_content)

print(f"✅ 已成功创建配置文件: {output_file}")
print(f"文件大小: {os.path.getsize(output_file)} bytes")
