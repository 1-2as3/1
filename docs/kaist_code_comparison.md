# KAIST 数据集使用代码对比

## 您的原始代码 vs 正确实现

### ❌ 原始代码（不兼容 MMDetection 3.x）

```python
import os, cv2
from mmdet.datasets import KAISTDataset

dataset = KAISTDataset(
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/Annotations/',  # ❌ 错误：应该是 ImageSets/*.txt
    img_prefix=dict(  # ❌ 错误：MMDetection 3.x 使用 data_prefix
        visible='C:/KAIST_processed/visible/', 
        infrared='C:/KAIST_processed/infrared/'
    )
)

print(f"样本总数: {len(dataset.img_infos)}")  # ❌ 错误：应该是 len(dataset)
for i in range(3):
    item = dataset[i]  # ❌ 错误：默认模式不返回配对数据
    vis, ir = item['visible'], item['infrared']
    both = cv2.hconcat([vis, ir])
    cv2.imwrite(f"sample_pair_{i}.jpg", both)
```

**主要问题**：
1. `ann_file` 应该指向 `ImageSets/train.txt`，不是 `Annotations/` 目录
2. MMDetection 3.x 使用 `data_prefix`，不是 `img_prefix`
3. 没有调用 `register_all_modules()`，会导致 registry 错误
4. 直接实例化类，应该使用 `DATASETS.build()`
5. 默认模式不返回配对数据，需要设置 `return_modality_pair=True`

---

### ✅ 正确实现（MMDetection 3.x 兼容）

```python
import os, cv2
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

# 步骤 1: 注册模块（必需！）
register_all_modules(init_default_scope=True)

# 步骤 2: 构建数据集配置
dataset = DATASETS.build(dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',  # ✅ 正确：指向样本列表
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),  # ✅ 正确：使用 data_prefix
    ann_subdir='Annotations',
    test_mode=True,
    return_modality_pair=True  # ✅ 启用双模态配对
))

print(f"样本总数: {len(dataset)}")  # ✅ 正确：直接使用 len()
for i in range(3):
    item = dataset[i]
    vis, ir = item['visible'], item['infrared']  # ✅ 正确：配对模式返回两个字段
    both = cv2.hconcat([vis, ir])
    cv2.imwrite(f"sample_pair_{i}.jpg", both)

print("✅ 已输出 3 对样本，可人工验证匹配正确。")
```

---

## 关键差异对照表

| 项目 | 原始代码 | 正确实现 | 说明 |
|------|---------|---------|------|
| **模块注册** | 无 | `register_all_modules()` | MMDetection 3.x 必需 |
| **实例化方式** | `KAISTDataset(...)` | `DATASETS.build(dict(...))` | 使用 registry 系统 |
| **ann_file** | `Annotations/` | `ImageSets/train.txt` | VOC 格式约定 |
| **data_prefix** | `img_prefix=dict(...)` | `data_prefix=dict(sub_data_root=...)` | API 变更 |
| **配对模式** | 无 | `return_modality_pair=True` | 启用双模态 |
| **样本数量** | `len(dataset.img_infos)` | `len(dataset)` | 标准 Python API |
| **返回格式** | 未定义 | `{'visible': arr, 'infrared': arr, ...}` | 明确的双模态格式 |

---

## 完整可运行示例

### 示例 1: 最简单的双模态可视化

```python
"""最简单的 KAIST 双模态测试"""
import os, cv2
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

# 注册模块
register_all_modules(init_default_scope=True)

# 创建临时小数据集（避免加载全部数据）
data_root = 'C:/KAIST_processed/'
temp_ann = os.path.join(data_root, "temp.txt")
with open(os.path.join(data_root, 'ImageSets/train.txt'), 'r') as f:
    with open(temp_ann, 'w') as fw:
        fw.writelines(f.readlines()[:5])  # 只取 5 个样本

try:
    # 构建数据集
    dataset = DATASETS.build(dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=temp_ann,
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=True,
        return_modality_pair=True
    ))
    
    print(f"样本总数: {len(dataset)}")
    
    # 可视化前 3 对
    for i in range(3):
        item = dataset[i]
        vis, ir = item['visible'], item['infrared']
        both = cv2.hconcat([vis, ir])
        cv2.imwrite(f"sample_pair_{i}.jpg", both)
        print(f"✅ sample_pair_{i}.jpg")
    
    print("✅ 已输出 3 对样本，可人工验证匹配正确。")
    
finally:
    os.remove(temp_ann)
```

### 示例 2: 带标注框的可视化

```python
"""带标注框的双模态可视化"""
import os, cv2
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

register_all_modules(init_default_scope=True)

data_root = 'C:/KAIST_processed/'
temp_ann = os.path.join(data_root, "temp.txt")
with open(os.path.join(data_root, 'ImageSets/train.txt'), 'r') as f:
    with open(temp_ann, 'w') as fw:
        fw.writelines(f.readlines()[:5])

try:
    dataset = DATASETS.build(dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=temp_ann,
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=True,
        return_modality_pair=True
    ))
    
    for i in range(3):
        item = dataset[i]
        vis = item['visible'].copy()
        ir = item['infrared'].copy()
        
        # 绘制标注框
        for inst in item['instances']:
            x1, y1, x2, y2 = map(int, inst['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(ir, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, 'person', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(ir, 'person', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        both = cv2.hconcat([vis, ir])
        cv2.imwrite(f"sample_pair_annotated_{i}.jpg", both)
        print(f"✅ sample_pair_annotated_{i}.jpg ({len(item['instances'])} persons)")
    
finally:
    os.remove(temp_ann)
```

### 示例 3: 单模态训练配置

```python
"""单模态训练配置（标准 MMDetection 流程）"""
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)

# 配置文件中的数据集定义
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='KAISTDataset',
        data_root='C:/KAIST_processed/',
        ann_file='C:/KAIST_processed/ImageSets/train.txt',
        data_prefix=dict(sub_data_root='C:/KAIST_processed'),
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

# 使用标准命令训练
# python tools/train.py configs/your_config.py
```

---

## 常见错误及解决方案

### 错误 1: `ModuleNotFoundError: No module named 'mmcv'`
**原因**: 未配置 Python 环境  
**解决**: 使用正确的 conda 环境
```bash
C:/Users/Xinyu/.conda/envs/py311/python.exe your_script.py
```

### 错误 2: `KeyError: 'PackDetInputs is not in the mmengine::transform registry'`
**原因**: 未调用 `register_all_modules()`  
**解决**: 在所有代码之前添加
```python
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
```

### 错误 3: `FileNotFoundError: ann_file not found`
**原因**: `ann_file` 路径错误  
**解决**: 必须使用绝对路径指向 `ImageSets/*.txt`
```python
ann_file='C:/KAIST_processed/ImageSets/train.txt'  # ✅ 正确
# 不是：
ann_file='C:/KAIST_processed/Annotations/'  # ❌ 错误
```

### 错误 4: `KeyError: 'visible' or 'infrared'`
**原因**: 未启用 `return_modality_pair=True`  
**解决**: 在配置中添加
```python
return_modality_pair=True  # 启用双模态配对
```

---

## 快速检查清单

使用 KAIST 数据集前，确保满足以下条件：

- [ ] 已调用 `register_all_modules(init_default_scope=True)`
- [ ] 使用 `DATASETS.build(dict(...))` 而不是直接实例化
- [ ] `ann_file` 指向 `ImageSets/train.txt`（绝对路径）
- [ ] 使用 `data_prefix=dict(sub_data_root='...')`
- [ ] 双模态模式需要 `return_modality_pair=True`
- [ ] 使用正确的 Python 环境（py311 conda）

---

## 相关文件

- **核心实现**: `mmdet/datasets/kaist_dataset.py`
- **完整文档**: `docs/kaist_dataset_guide.md`
- **测试脚本**: `test_kaist_paired_modality.py`
- **简单示例**: `simple_kaist_test.py`
- **配置示例**: `configs/kaist/kaist_dataset_usage.py`

---

**更新日期**: 2025-11-07  
**MMDetection 版本**: 3.x
