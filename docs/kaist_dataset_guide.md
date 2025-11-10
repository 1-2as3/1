# KAIST 数据集使用指南

## 概述

`KAISTDataset` 是基于 MMDetection VOCDataset 扩展的多模态行人检测数据集类，支持：
- ✅ **单模态加载**：标准 VOC 格式训练流程
- ✅ **双模态配对**：同时返回可见光 + 红外图像对
- ✅ **动态路径解析**：自动识别 visible/infrared 子目录
- ✅ **统一标注格式**：VOC XML 格式

## 目录结构

```
C:/KAIST_processed/
├── Annotations/          # VOC XML 标注文件
│   ├── set00_V000_visible_I01216.xml
│   ├── set00_V000_lwir_I01216.xml
│   └── ...
├── visible/              # 可见光图像
│   ├── set00_V000_visible_I01216.jpg
│   └── ...
├── infrared/             # 红外图像
│   ├── set00_V000_lwir_I01216.jpg
│   └── ...
└── ImageSets/
    ├── train.txt         # 训练集样本ID列表
    ├── val.txt
    └── test.txt
```

## 核心功能

### 1. 单模态模式（默认）

**配置示例**:
```python
dataset = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    ann_subdir='Annotations',
    test_mode=False,
    return_modality_pair=False,  # 单模态模式（默认）
    pipeline=[...]
)
```

**特点**:
- 根据样本ID自动选择 visible/ 或 infrared/ 子目录
- 支持标准 MMDetection pipeline
- 可直接用于 `tools/train.py` 训练
- 返回格式：标准 MMDetection 数据格式（含 data_samples）

### 2. 双模态配对模式

**配置示例**:
```python
dataset = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    ann_subdir='Annotations',
    test_mode=True,  # 跳过 pipeline
    return_modality_pair=True,  # 启用双模态配对
)
```

**返回数据格式**:
```python
{
    'visible': np.ndarray,        # (H, W, 3) BGR 格式
    'infrared': np.ndarray,       # (H, W, 3) BGR 格式
    'visible_path': str,          # 完整路径
    'infrared_path': str,         # 完整路径
    'base_id': str,               # 去除模态关键词的基础ID
    'instances': List[dict],      # 标注实例（bbox, labels）
    'metainfo': {
        'modality': 'paired',
        'img_shape': (H, W),
        'ori_shape': (H, W)
    }
}
```

**特点**:
- 自动配对同一场景的 visible + infrared 图像
- 验证两个模态的文件都存在
- 保证两个图像尺寸一致
- 需要自定义训练循环处理（不使用标准 pipeline）

## 样本ID命名规则

KAIST 数据集使用以下命名约定：

| 模态 | 样本ID 示例 | 对应图像路径 |
|------|------------|-------------|
| 可见光 | `set00_V000_visible_I01216` | `visible/set00_V000_visible_I01216.jpg` |
| 红外 | `set00_V000_lwir_I01216` | `infrared/set00_V000_lwir_I01216.jpg` |
| Base ID | `set00_V000_I01216` | 用于配对查找 |

**Base ID 提取逻辑**:
```python
# 去除模态关键词
base_id = img_id.replace('_visible', '').replace('_lwir', '')
# set00_V000_visible_I01216 → set00_V000_I01216
# set00_V000_lwir_I01216 → set00_V000_I01216
```

## Python API 使用示例

### 示例 1: 单模态加载

```python
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

# 注册模块（必须！）
register_all_modules(init_default_scope=True)

# 构建数据集
cfg = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    ann_subdir='Annotations',
    test_mode=False,
    return_modality_pair=False,
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_label=True),
        dict(type='Resize', scale=(640, 512), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ]
)

dataset = DATASETS.build(cfg)

# 加载样本
for i in range(len(dataset)):
    data = dataset[i]
    print(f"Sample {i}:")
    print(f"  inputs.shape: {data['inputs'].shape}")
    print(f"  modality: {data['data_samples'].metainfo['modality']}")
    print(f"  gt_instances: {len(data['data_samples'].gt_instances)}")
```

### 示例 2: 双模态配对加载

```python
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS
import cv2
import numpy as np

# 注册模块
register_all_modules(init_default_scope=True)

# 构建数据集（双模态模式）
cfg = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    ann_subdir='Annotations',
    test_mode=True,
    return_modality_pair=True,  # 关键：启用配对模式
)

dataset = DATASETS.build(cfg)

# 加载配对数据
for i in range(len(dataset)):
    data = dataset[i]
    
    # 访问双模态图像
    visible_img = data['visible']
    infrared_img = data['infrared']
    
    print(f"Sample {i}:")
    print(f"  base_id: {data['base_id']}")
    print(f"  visible shape: {visible_img.shape}")
    print(f"  infrared shape: {infrared_img.shape}")
    print(f"  instances: {len(data['instances'])}")
    
    # 可视化配对图像
    combined = np.hstack([visible_img, infrared_img])
    cv2.imshow('Paired Images', combined)
    cv2.waitKey(500)
```

### 示例 3: 自定义多模态训练循环

```python
import torch
from torch.utils.data import DataLoader
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

register_all_modules(init_default_scope=True)

# 自定义 collate 函数
def paired_collate_fn(batch):
    """处理双模态批次数据"""
    visible_imgs = [item['visible'] for item in batch]
    infrared_imgs = [item['infrared'] for item in batch]
    instances = [item['instances'] for item in batch]
    
    # 转换为 tensor
    visible_batch = torch.stack([torch.from_numpy(img).permute(2,0,1) for img in visible_imgs])
    infrared_batch = torch.stack([torch.from_numpy(img).permute(2,0,1) for img in infrared_imgs])
    
    return {
        'visible': visible_batch,
        'infrared': infrared_batch,
        'instances': instances
    }

# 构建数据集
cfg = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    ann_subdir='Annotations',
    test_mode=True,
    return_modality_pair=True,
)

dataset = DATASETS.build(cfg)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=paired_collate_fn
)

# 训练循环
for epoch in range(10):
    for batch_idx, batch in enumerate(dataloader):
        visible_imgs = batch['visible']    # (B, 3, H, W)
        infrared_imgs = batch['infrared']  # (B, 3, H, W)
        
        # 多模态融合
        fused_features = your_fusion_model(visible_imgs, infrared_imgs)
        
        # 继续训练...
```

## 配置文件集成

### stage2_kaist_domain_ft.py（单模态微调）

```python
_base_ = ['../_base_/default_runtime.py']

data_root = 'C:/KAIST_processed/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=f'{data_root}/ImageSets/train.txt',
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=False,
        return_modality_pair=False,  # 单模态训练
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=f'{data_root}/ImageSets/val.txt',
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=True,
        return_modality_pair=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    )
)
```

## 常见问题

### Q1: 为什么需要 `register_all_modules()`？
A: MMDetection 3.x 使用 registry 系统管理模块。必须先注册才能构建数据集和 transforms。

### Q2: ann_file 必须是绝对路径吗？
A: 是的。VOCDataset 不支持相对路径，必须使用完整绝对路径。

### Q3: 双模态模式下如何处理缺失的配对文件？
A: 代码会自动检查并抛出 `FileNotFoundError`，明确指出缺失的文件路径。

### Q4: 可以混合使用 visible 和 infrared 样本吗？
A: 可以。单模态模式会自动识别样本ID中的模态关键词，动态选择子目录。

### Q5: 双模态模式能用标准 pipeline 吗？
A: 不能。双模态模式返回原始图像对，需要自定义处理逻辑。标准 pipeline 仅支持单张图像。

## 测试验证

运行测试脚本验证功能：

```bash
# 测试单模态 + 双模态功能
python test_kaist_paired_modality.py

# 预期输出：
# - 单模态模式：成功加载5个样本
# - 双模态模式：成功加载并保存配对图像
# - 保存 kaist_paired_sample_*.jpg 到当前目录
```

## 下一步

1. **单模态预训练**：使用 LLVIP 或 KAIST 单模态数据预训练检测器
2. **双模态融合**：实现自定义融合网络处理配对数据
3. **领域自适应**：LLVIP → KAIST 跨数据集迁移学习

---

**更新日期**: 2025-11-07  
**版本**: v1.0  
**维护者**: MMDetection Custom Dataset Team
