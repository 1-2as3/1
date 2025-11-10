from mmengine.config import read_base

# ========= 使用 Lazy Config 模式继承基础配置 =========
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.voc0712 import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *


# 设置全局训练配置选项
train_cfg = dict(
    use_macl=True,  # 是否启用 MACL
    use_msp=True,   # 是否启用 MSP
    use_dhn=True,   # 是否启用 DHN
    loss_macl_weight=0.2  # MACL 损失权重
)

# Merge with base `model` imported via read_base()
# Update only the fields we need so base config stays intact.
model.setdefault('backbone', {}).update(dict(type='ResNet', depth=50))
model.setdefault('neck', {}).update(dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    use_msp=train_cfg['use_msp'],
    msp_module=dict(
        type='MSPReweight',
        channels=256,
    ) if train_cfg['use_msp'] else None
))

# DHN sampler configuration (used by MACL)
dhn_config = dict(
    type='DHNSampler',
    queue_size=8192,
    momentum=0.99,
    top_k=256  # number of hard negatives to sample
) if train_cfg['use_dhn'] else None

# Update roi_head without overwriting other important sub-keys from base
model.setdefault('roi_head', {}).update(dict(
    type='StandardRoIHead',
    use_macl=train_cfg['use_macl'],
    macl_head=dict(
        type='MACLHead',
        in_dim=256,
        proj_dim=128,
        temperature=0.07,
        dhn_sampler=dhn_config,  # conditionally attach DHN
        loss_weight=train_cfg['loss_macl_weight']  # scale MACL loss
    ) if train_cfg['use_macl'] else None
))

# ===============================
# 兼容 MMDetection 3.x 配置结构
# ===============================
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='LLVIPDataset',
        ann_file='C:/LLVIP/LLVIP/Annotations/train.txt',
        data_prefix=dict(
            visible='C:/LLVIP/LLVIP/visible/train/',
            infrared='C:/LLVIP/LLVIP/infrared/train/'
        )
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='LLVIPDataset',
        ann_file='C:/LLVIP/LLVIP/Annotations/test.txt',
        data_prefix=dict(
            visible='C:/LLVIP/LLVIP/visible/test/',
            infrared='C:/LLVIP/LLVIP/infrared/test/'
        )
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

# ⚠️ 定义完整的测试数据加载器配置
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='LLVIPDataset',
        data_root='C:/LLVIP/LLVIP/',
        ann_file='C:/LLVIP/LLVIP/Annotations/test.txt',
        data_prefix=dict(
            visible='C:/LLVIP/LLVIP/visible/test/',
            infrared='C:/LLVIP/LLVIP/infrared/test/'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

# 添加自定义 hooks 用于参数监控
custom_hooks = [
    dict(
        type='ParameterMonitorHook',
        log_dir='runs/macl_msp_dhn',
        interval=1,
        by_epoch=True
    ),
    dict(
        type='UpdateEpochHook'  # 更新 epoch 信息，用于动态损失权重
    )
]

# 优化器配置：冻结 backbone 前两层（阶段二）
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone.layer1': dict(lr_mult=0.0, decay_mult=0.0)
        }
    )
)

