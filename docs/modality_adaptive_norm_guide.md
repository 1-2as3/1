# ModalityAdaptiveNorm 集成指南

## 概述
ModalityAdaptiveNorm 为不同模态（可见光/红外）使用独立的批归一化层，允许网络学习模态特定的特征统计信息。

## 核心思想
- 可见光和红外图像具有不同的统计特性
- 使用共享的 BN 层可能限制模型性能
- 为每个模态使用独立的 BN 层可以更好地适应各自的分布

## 实现方式

### 方式1: 使用 ModalityAdaptiveNorm 模块

```python
from mmdet.models.macldhnmsp import ModalityAdaptiveNorm

# 创建模态自适应 BN 层
modality_bn = ModalityAdaptiveNorm(channels=64)

# 在前向传播中使用
x = modality_bn(x, modality='visible')  # 或 'infrared'
```

### 方式2: 使用 ModalityAdaptiveResNet 包装器

```python
# 在配置文件中
model = dict(
    backbone=dict(
        type='ModalityAdaptiveResNet',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            ...
        ),
        use_modality_bn=True,
        modality_bn_layers=[0, 1]  # 前两层使用模态自适应 BN
    )
)
```

### 方式3: 继承并修改 ResNet

```python
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
from mmdet.models.macldhnmsp import ModalityAdaptiveNorm

@MODELS.register_module()
class ResNetWithModalityBN(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换 bn1 为模态自适应 BN
        self.modality_bn1 = ModalityAdaptiveNorm(64)
        
        # 为 layer1 的输出添加模态自适应 BN
        self.modality_bn_layer1 = ModalityAdaptiveNorm(256)  # ResNet50
    
    def forward(self, x, modality=None):
        # Stem
        x = self.conv1(x)
        x = self.modality_bn1(x, modality)  # 使用模态自适应 BN
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1
        x = self.layer1(x)
        x = self.modality_bn_layer1(x, modality)  # 使用模态自适应 BN
        
        # Layer 2-4 (标准实现)
        outs = []
        outs.append(x)
        
        x = self.layer2(x)
        outs.append(x)
        
        x = self.layer3(x)
        outs.append(x)
        
        x = self.layer4(x)
        outs.append(x)
        
        return tuple(outs)
```

## 数据集适配

### LLVIPDataset 修改示例

```python
# mmdet/datasets/llvip_dataset.py

def __getitem__(self, idx):
    data = super().__getitem__(idx)
    
    # 从图像路径判断模态
    img_path = data['data_samples'].img_path
    if 'visible' in img_path.lower():
        modality = 'visible'
    elif 'infrared' in img_path.lower():
        modality = 'infrared'
    else:
        modality = 'unknown'
    
    # 添加到 metainfo
    data['data_samples'].metainfo['modality'] = modality
    
    # 也可以添加到顶层，便于 backbone 访问
    data['modality'] = modality
    
    return data
```

### Detector 修改示例

```python
# 在 detector 的 extract_feat 方法中传递模态信息

def extract_feat(self, batch_inputs, batch_data_samples):
    # 提取模态信息
    modalities = [sample.metainfo.get('modality', None) 
                  for sample in batch_data_samples]
    
    # 传递给 backbone
    x = self.backbone(batch_inputs, modality=modalities)
    
    if self.with_neck:
        x = self.neck(x)
    
    return x
```

## 训练策略

### 1. 单模态批次（推荐初期训练）
```python
# 每个 batch 只包含一种模态
train_dataloader = dict(
    batch_size=4,
    # 可以使用自定义 sampler 确保每个 batch 模态一致
)
```

优点：
- BN 统计更稳定
- 避免小 batch 问题

### 2. 混合模态批次（推荐后期微调）
```python
# 每个 batch 包含不同模态
train_dataloader = dict(
    batch_size=4,  # 例如 2 个可见光 + 2 个红外
    sampler=dict(type='BalancedModalitySampler')
)
```

优点：
- 更好的模态对齐
- 提高泛化能力

### 3. 学习率调整
```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9),
    paramwise_cfg=dict(
        custom_keys={
            # 冻结或降低前两层的学习率
            'backbone.conv1': dict(lr_mult=0.1),
            'backbone.layer1': dict(lr_mult=0.1),
            
            # 模态 BN 层使用正常学习率
            'modality_bn': dict(lr_mult=1.0),
        }
    )
)
```

## 性能优化

### 1. 内存优化
- 使用单模态批次可以减少内存占用
- 考虑使用 gradient checkpointing

### 2. 速度优化
- 模态信息应该尽早确定（在数据加载时）
- 避免在 forward 中进行复杂的模态判断逻辑

### 3. 效果优化
- 初期训练：使用单模态批次，让 BN 层充分学习各模态统计
- 中期训练：逐渐引入混合批次
- 后期微调：使用混合批次，提高模态间对齐

## 实验建议

### 对照实验
1. **Baseline**: 标准 ResNet（共享 BN）
2. **Modality BN - Layer0**: 只在 stem 使用模态 BN
3. **Modality BN - Layer0+1**: 在前两层使用模态 BN（推荐）
4. **Modality BN - All**: 所有层使用模态 BN

### 评估指标
- 可见光图像 mAP
- 红外图像 mAP
- 整体 mAP
- 跨模态泛化能力

## 故障排除

### 问题1: BN 统计不稳定
**症状**: 训练损失震荡，验证性能差
**原因**: 混合模态批次中每个模态样本太少
**解决**:
- 增加 batch size
- 使用单模态批次
- 调整 BN momentum

### 问题2: 性能没有提升
**症状**: 使用模态 BN 后性能反而下降
**原因**: 可能的原因包括：
- BN 层放置位置不当
- 学习率不合适
- 训练不充分
**解决**:
- 只在浅层（layer0, layer1）使用模态 BN
- 延长训练时间
- 尝试不同的学习率

### 问题3: 推理时模态信息缺失
**症状**: 测试时无法确定模态
**原因**: 测试数据没有模态标注
**解决**:
```python
# 方法1: 从文件名推断
modality = 'visible' if 'vis' in img_path else 'infrared'

# 方法2: 使用默认 BN
x = modality_bn(x, modality=None)  # 使用 bn_default

# 方法3: 训练轻量级模态分类器
```

## 参考文献
- Domain-Specific Batch Normalization
- Cross-Modal Learning with BN
- Multi-Modal Feature Fusion
