from mmengine.config import read_base

with read_base():
    from .faster_rcnn_r50_fpn_macl_msp_dhn import *

# 使用 Modality-Adaptive Normalization 的配置

# 方法1: 使用包装的 ModalityAdaptiveResNet
model.update(dict(
    backbone=dict(
        type='ModalityAdaptiveResNet',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        use_modality_bn=True,
        modality_bn_layers=[0, 1]  # 在前两层使用模态自适应 BN
    )
))

# 注意：使用 ModalityAdaptiveResNet 时，需要确保数据样本中包含模态信息
# 在数据加载时添加模态标注，例如：
# data['modality'] = 'visible'  或  data['modality'] = 'infrared'

# 方法2: 直接在现有 ResNet 中插入 ModalityAdaptiveNorm
# 这需要修改 ResNet 的 forward 方法，通常通过继承实现

# 训练时的注意事项：
# 1. 确保数据集的 __getitem__ 方法返回模态信息
# 2. 在 forward 时传递模态信息到 backbone
# 3. 可以使用混合批次（同时包含可见光和红外图像）

# 示例数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='LLVIPDataset',
        ann_file='C:/LLVIP/LLVIP/Annotations/train.txt',
        data_prefix=dict(
            visible='C:/LLVIP/LLVIP/visible/train/',
            infrared='C:/LLVIP/LLVIP/infrared/train/'
        ),
        # LLVIPDataset 应该在 __getitem__ 中返回模态信息
        # 例如: data_samples.metainfo['modality'] = 'visible' or 'infrared'
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

# 优化器配置 - 模态自适应 BN 层使用独立学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0.1, decay_mult=0.1),  # 较低学习率
            'backbone.layer1': dict(lr_mult=0.1, decay_mult=0.1),  # 较低学习率
            'modality_bns': dict(lr_mult=1.0, decay_mult=1.0),     # 正常学习率
        }
    )
)
