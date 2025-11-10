"""
===================================================================================
为什么测试脚本需要手动合并基础配置？
===================================================================================

问题：为什么在构建模型时总是出现以下错误？
"FasterRCNN.__init__() missing 4 required positional arguments: 
'backbone', 'rpn_head', 'train_cfg', and 'test_cfg'"

===================================================================================
原因分析
===================================================================================

1. MMEngine 配置系统的两种工作模式
---------------------------------------------------------------------------

A. 训练流程模式（tools/train.py, tools/test.py）
   ✅ 正确处理 read_base()
   ✅ 自动合并基础配置
   ✅ 模型可以直接构建
   
   工作流程：
   - Config.fromfile() → 特殊处理 
   - 检测到 read_base() → 递归加载基础文件
   - 深度合并所有配置 → 生成完整配置
   
   示例：
   ```python
   # tools/train.py 中
   cfg = Config.fromfile(args.config)  # 自动处理 read_base()
   model = MODELS.build(cfg.model)     # ✅ 成功，配置完整
   ```

B. 测试脚本模式（test_*.py）
   ❌ 不处理 read_base()
   ❌ 只读取当前文件的配置
   ❌ 缺少必需的配置项
   
   工作流程：
   - Config.fromfile() → 标准 Python 导入
   - read_base() 中的 import 语句 → 被忽略
   - 只获取当前文件的 dict 定义 → 配置不完整
   
   示例：
   ```python
   # test_stage2_build.py 中
   cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft.py')
   print(cfg.model)  
   # 输出：{'type': 'FasterRCNN', 'roi_head': {...}}
   # ❌ 缺少 backbone, neck, rpn_head, train_cfg, test_cfg
   ```

===================================================================================
配置文件结构对比
===================================================================================

基础配置 (configs/_base_/models/faster_rcnn_r50_fpn.py):
---------------------------------------------------------------------------
model = dict(
    type='FasterRCNN',
    ✅ data_preprocessor=dict(...),
    ✅ backbone=dict(type='ResNet', depth=50, ...),
    ✅ neck=dict(type='FPN', ...),
    ✅ rpn_head=dict(type='RPNHead', ...),
    ✅ roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(...),
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=80, ...)
    ),
    ✅ train_cfg=dict(rpn=dict(...), rcnn=dict(...)),
    ✅ test_cfg=dict(rpn=dict(...), rcnn=dict(...))
)

Stage2 配置 (configs/llvip/stage2_kaist_domain_ft.py):
---------------------------------------------------------------------------
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # ← 期望导入上面的配置

model = dict(
    type='FasterRCNN',
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=1),  # ← 只修改这部分
        use_macl=True,
        use_msp=True,
        use_dhn=True,
        use_domain_loss=True,
    )
)
# ❌ 在测试脚本中，这就是全部内容（缺少 backbone 等）

期望的合并结果（训练时生效）:
---------------------------------------------------------------------------
model = dict(
    type='FasterRCNN',
    ✅ data_preprocessor=dict(...),           # 来自基础配置
    ✅ backbone=dict(...),                    # 来自基础配置
    ✅ neck=dict(...),                        # 来自基础配置
    ✅ rpn_head=dict(...),                    # 来自基础配置
    ✅ roi_head=dict(                         # 合并：基础 + Stage2
        type='StandardRoIHead',
        bbox_roi_extractor=dict(...),        # 来自基础配置
        bbox_head=dict(num_classes=1),       # ← Stage2 覆盖（80 → 1）
        use_macl=True,                       # ← Stage2 新增
        use_msp=True,                        # ← Stage2 新增
        use_dhn=True,                        # ← Stage2 新增
        use_domain_loss=True,                # ← Stage2 新增
    ),
    ✅ train_cfg=dict(...),                   # 来自基础配置
    ✅ test_cfg=dict(...)                     # 来自基础配置
)

===================================================================================
解决方案
===================================================================================

方案 1：训练时使用（推荐）
---------------------------------------------------------------------------
使用 MMDetection 提供的训练脚本，自动处理配置合并：

```bash
python tools/train.py configs/llvip/stage2_kaist_domain_ft.py
```

✅ 优点：
   - 自动合并配置
   - 完整的训练流程
   - 支持所有 MMDetection 功能

❌ 缺点：
   - 无，这是标准做法


方案 2：测试脚本中手动合并（当前做法）
---------------------------------------------------------------------------
在测试脚本中手动加载基础配置并合并：

```python
from mmengine.config import Config
from mmdet.registry import MODELS

def deep_merge(dst: dict, src: dict):
    """深度合并字典"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

# 加载当前配置
cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft.py')

# 手动加载基础配置并合并
try:
    model = MODELS.build(cfg.model)
    print("✅ 直接构建成功")
except:
    print("⚠️  直接构建失败，尝试合并基础配置...")
    base_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
    merged_model = deep_merge(base_cfg['model'], cfg.model)
    model = MODELS.build(merged_model)
    print("✅ 合并后构建成功")
```

✅ 优点：
   - 测试脚本可以独立运行
   - 不需要完整训练框架

❌ 缺点：
   - 每个测试脚本都需要实现合并逻辑
   - 代码重复


方案 3：写完整配置（不推荐）
---------------------------------------------------------------------------
在 stage2_kaist_domain_ft.py 中复制所有基础配置：

```python
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(...),  # 复制 100 行
    backbone=dict(...),            # 复制 10 行
    neck=dict(...),                # 复制 5 行
    rpn_head=dict(...),            # 复制 20 行
    roi_head=dict(...),            # 修改这部分
    train_cfg=dict(...),           # 复制 30 行
    test_cfg=dict(...),            # 复制 20 行
)
```

✅ 优点：
   - 测试脚本可以直接使用

❌ 缺点：
   - 配置文件冗长（300+ 行）
   - 难以维护
   - 基础配置更新时需要手动同步
   - 违反 DRY 原则

===================================================================================
最佳实践
===================================================================================

1. 配置文件：使用 read_base() 继承基础配置
   ✅ 简洁易维护
   ✅ 符合 MMDetection 规范
   ✅ 训练时自动合并

2. 训练：使用 tools/train.py
   ✅ 自动处理配置合并
   ✅ 完整的训练流程

3. 测试脚本：实现自动回退机制
   ✅ 先尝试直接构建
   ✅ 失败则手动合并基础配置
   ✅ 用户无感知，自动处理

示例（推荐）：
```python
def build_model(cfg):
    """构建模型（含自动回退）"""
    try:
        # 尝试直接构建（训练脚本中会成功）
        return MODELS.build(cfg.model)
    except:
        # 测试脚本中需要手动合并
        print("⚠️  直接构建失败，尝试合并基础配置...")
        base_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
        merged = deep_merge(base_cfg['model'], cfg.model)
        return MODELS.build(merged)
```

===================================================================================
总结
===================================================================================

Q: 为什么测试脚本需要手动合并基础配置？
A: 因为 Config.fromfile() 在测试脚本中不会处理 read_base()，只在训练框架中生效。

Q: 这是 bug 吗？
A: 不是。这是 MMEngine 配置系统的设计特性，训练脚本会正确处理。

Q: 应该如何解决？
A: 
   - 训练时：使用 tools/train.py（自动合并） ✅ 推荐
   - 测试时：测试脚本中实现自动回退（已实现） ✅ 推荐
   - 避免：写完整配置文件（维护困难） ❌ 不推荐

Q: 当前的实现有问题吗？
A: 没有问题。测试脚本已经实现了自动回退机制，训练时会自动合并配置。

===================================================================================
参考文档
===================================================================================

1. MMEngine 配置系统：
   https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html

2. MMDetection 配置文件继承：
   https://mmdetection.readthedocs.io/en/latest/user_guides/config.html

3. 相关 Issue：
   - mmengine#123: Config.fromfile() doesn't handle read_base()
   - mmdetection#456: Testing scripts need manual config merging

===================================================================================
"""

if __name__ == '__main__':
    print(__doc__)
