from mmengine.config import read_base

with read_base():
    from .faster_rcnn_r50_fpn_macl_msp_dhn import *

# ============ 三阶段训练损失权重配置 ============

# 阶段一：基础检测 + MACL（仅 LLVIP）
# 推荐 epoch: 0-30
# lambda1=1.0, lambda2=0.0, lambda3=0.0

# 阶段二：+ DHN 困难负样本（LLVIP + 小规模其他数据集）
# 推荐 epoch: 31-60
# lambda1=1.0, lambda2=0.5, lambda3=0.0

# 阶段三：+ 域对齐（多数据集联合训练）
# 推荐 epoch: 61-100
# lambda1=1.0, lambda2=0.5, lambda3=0.1（动态衰减）

# 更新 ROI Head 配置，添加损失权重参数
model.roi_head.update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_domain_alignment=True,  # 阶段三启用
    
    # 三阶段损失权重
    lambda1=1.0,  # MACL 损失权重
    lambda2=0.5,  # DHN 损失权重
    lambda3=0.1,  # 域对齐损失权重（会动态衰减）
    
    macl_head=dict(
        type='MACLHead',
        in_dim=256,
        proj_dim=128,
        tau=0.07,
        use_bn=True,
        use_dhn=True  # 阶段二启用 DHN
    ),
    
    domain_classifier=dict(
        type='DomainClassifier',
        in_dim=256,  # 根据特征维度调整
        num_domains=3,  # LLVIP, KAIST, M3FD
        hidden_dim=128
    )
))

# 多数据集配置（阶段三）
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='LLVIPDataset',
                ann_file='C:/LLVIP/LLVIP/Annotations/train.txt',
                data_prefix=dict(
                    visible='C:/LLVIP/LLVIP/visible/train/',
                    infrared='C:/LLVIP/LLVIP/infrared/train/'
                ),
                metainfo=dict(domain_id=0, domain_name='LLVIP')
            ),
            dict(
                type='KAISTDataset',
                data_root='path/to/KAIST',
                ann_file='path/to/KAIST/annotations/train.txt',
                data_prefix=dict(
                    visible='path/to/KAIST/visible/train',
                    infrared='path/to/KAIST/lwir/train'
                ),
                metainfo=dict(domain_id=1, domain_name='KAIST')
            ),
            dict(
                type='M3FDDataset',
                data_root='path/to/M3FD',
                ann_file='path/to/M3FD/annotations/train.txt',
                data_prefix=dict(
                    visible='path/to/M3FD/visible/train',
                    infrared='path/to/M3FD/infrared/train'
                ),
                metainfo=dict(domain_id=2, domain_name='M3FD')
            )
        ]
    ),
    sampler=dict(type='BalancedModalitySampler')
)

# 自定义 Hooks
custom_hooks = [
    dict(
        type='ParameterMonitorHook',
        log_dir='runs/three_stage_training',
        interval=1,
        by_epoch=True
    ),
    dict(
        type='UpdateEpochHook'  # 必需：更新 epoch 信息用于动态权重衰减
    ),
    dict(
        type='DomainAdaptationHook',
        initial_lambda=0.0,
        final_lambda=1.0,
        schedule='exp'
    )
]

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            # 冻结 backbone 前两层（阶段二/三）
            'backbone.layer0': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone.layer1': dict(lr_mult=0.0, decay_mult=0.0),
            
            # MACL 和域分类器使用正常学习率
            'macl_head': dict(lr_mult=1.0),
            'domain_classifier': dict(lr_mult=10.0)  # 域分类器更高学习率
        }
    )
)

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  # 总训练轮数
    val_interval=5   # 每 5 个 epoch 验证一次
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500  # warmup
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[30, 60, 90],  # 在阶段切换点降低学习率
        gamma=0.1
    )
]

# 日志配置
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=5,
        save_best='auto'
    )
)
