from mmengine.config import read_base

with read_base():
    from .faster_rcnn_r50_fpn_macl_msp_dhn import *

# 阶段二配置：添加域对齐模块
# 假设有 3 个域：LLVIP (id=0), KAIST (id=1), M3FD (id=2)

# 更新模型配置，添加域分类器
model.setdefault('roi_head', {}).update(dict(
    type='StandardRoIHead',
    use_macl=True,
    use_domain_alignment=True,  # 启用域对齐
    domain_classifier=dict(
        type='DomainClassifier',
        in_dim=256,
        num_domains=3,  # LLVIP, KAIST, M3FD
        hidden_dim=128
    ),
    domain_loss_weight=0.1,  # 域对齐损失权重
    macl_head=dict(
        type='MACLHead',
        in_dim=256,
        proj_dim=128,
        tau=0.07,
        use_bn=True,
        use_dhn=True
    )
))

# 训练数据加载器配置
# 需要为每个样本添加 domain_id 标注
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
                # 为 LLVIP 数据集添加域标识
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

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone.layer1': dict(lr_mult=0.0, decay_mult=0.0),
            # 域分类器使用更高的学习率
            'domain_classifier': dict(lr_mult=10.0)
        }
    )
)

# 学习率调度器 - 域对齐的 lambda 参数随训练进度增加
# lambda = 2 / (1 + exp(-10 * progress)) - 1
custom_hooks = [
    dict(
        type='ParameterMonitorHook',
        log_dir='runs/macl_msp_dhn_domain',
        interval=1,
        by_epoch=True
    ),
    # 域适应 Hook：动态调整 lambda 参数
    dict(
        type='DomainAdaptationHook',
        initial_lambda=0.0,
        final_lambda=1.0,
        schedule='exp'  # 使用指数调度
    )
]
