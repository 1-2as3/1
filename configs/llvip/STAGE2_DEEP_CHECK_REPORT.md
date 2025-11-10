# Stage2 配置深度检查报告

## 🔴 严重问题（3个）

### 1. ❌ Backbone 未正确冻结
**问题**：Backbone 有 23,282,688 个可训练参数（应该全部冻结）
- 总参数：23.51M
- 可训练：23.28M ❌
- 冻结：0.23M（仅 < 1%）

**原因**：`paramwise_cfg` 中的 `lr_mult=0.0` **只是设置学习率为0，并不真正冻结参数**！

**影响**：
- 虽然学习率为0，但参数仍然参与前向和反向传播
- 占用不必要的显存和计算资源
- 可能因数值误差导致微小更新
- 违背 Stage2 "冻结 backbone" 的设计原则

**修复方案**：
```python
# 在模型构建后显式冻结
for name, param in model.named_parameters():
    if 'backbone' in name:
        param.requires_grad = False
```

---

### 2. ❌ 预训练权重文件不存在
**问题**：`load_from = './work_dirs/stage1_llvip_pretrain/epoch_latest.pth'` 文件不存在

**影响**：
- Stage2 训练将从头开始（ImageNet 预训练），**而非从 Stage1 加载**
- 完全失去 Stage1 的 LLVIP 领域知识
- Stage2 的整个设计前提被打破

**修复方案**：
1. 先运行 Stage1 训练生成 `epoch_latest.pth`
2. 或使用已有的 checkpoint 路径
3. 或使用 MMDetection 官方的 ResNet50 预训练权重

---

### 3. ❌ 前向传播测试失败
**问题**：`AssertionError: assert img_meta.get('scale_factor') is not None`

**原因**：模拟的 `data_samples` 缺少必需的 `scale_factor` 元数据

**影响**：
- 推理阶段会失败
- 实际训练中可能也会遇到此问题（如果 pipeline 配置不当）

**修复方案**：
确保 data_samples 包含所有必需字段：
```python
ds.set_metainfo({
    'img_shape': (512, 640),
    'ori_shape': (512, 640),
    'scale_factor': (1.0, 1.0),
    'modality': 'infrared'
})
```

---

## 📊 详细参数统计

### 总体
- **总参数**：41.40M
- **可训练**：41.17M
- **冻结**：0.23M

### 各模块分布
| 模块 | 总参数 | 可训练 | 冻结 | 状态 |
|------|--------|--------|------|------|
| Backbone | 23.51M | 23.28M ❌ | 0.23M | **未正确冻结** |
| Neck | 3.34M | 3.34M ✓ | 0M | 正常 |
| RPN | 0.59M | 0.59M ✓ | 0M | 正常 |
| RoI Head | 13.90M | 13.90M ✓ | 0M | 正常 |
| MSP | 0.0005M | 0.0005M ✓ | 0M | 正常（515个参数） |
| MACL | 0.05M | 0.05M ✓ | 0M | 正常（49,729个参数） |

### 关键可训练参数验证
✓ `neck.msp_module.alpha` (1 参数，初始值 0.5000)
✓ `roi_head.macl_head.tau` (1 参数，初始值 0.0700)
✓ `roi_head.macl_head.proj.0.weight` (32,768 参数)
✓ `roi_head.macl_head.proj.3.weight` 

---

## ⚙️ 配置状态

### 学习率与优化器
- 基础学习率：0.0003 ✓
- Backbone lr_mult：0.0（但参数未真正冻结 ❌）
- 训练轮数：12 ✓

### 损失权重
- lambda1 (MACL)：1.0 ✓
- lambda2 (DHN)：0.5 ✓
- lambda3 (Domain)：0.1 ✓

### 数据集配置
- Train：C:/KAIST_processed/ ✓
- Val：C:/KAIST_processed/ ✓
- Test：C:/KAIST_processed/ ✓
- return_modality_pair：False（所有数据集） ✓

---

## 🎯 修复优先级

### 🔴 **必须立即修复**（阻塞训练）

1. **修复 Backbone 冻结**
   - 影响：显存、计算效率、设计原则
   - 操作：在配置中添加真正的参数冻结

2. **提供 Stage1 预训练权重**
   - 影响：Stage2 整个设计前提
   - 操作：训练 Stage1 或使用替代权重

### 🟡 **建议修复**（提升稳定性）

3. **修复 data_samples 元数据**
   - 影响：推理阶段可能失败
   - 操作：确保 pipeline 配置完整

---

## 📝 修复代码示例

### 1. 正确冻结 Backbone

在 `configs/llvip/stage2_kaist_domain_ft.py` 中添加：

```python
# 方案 A：在配置中使用自定义 hook
custom_hooks = [
    dict(
        type='CustomHook',
        priority=49,  # 在 optimizer hook 之前
        callback=lambda runner: freeze_backbone(runner.model)
    )
]

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
```

### 2. 使用官方预训练权重（临时方案）

```python
# 如果 Stage1 权重不可用，使用 MMDet 官方权重
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
```

### 3. 完整的 data_samples 元数据

```python
# 在 KAISTDataset 或 pipeline 中确保包含
metainfo = dict(
    img_shape=(512, 640),
    ori_shape=(512, 640),
    scale_factor=(1.0, 1.0),
    pad_shape=(512, 640),
    modality='infrared'
)
```

---

## ✅ 已验证正常的部分

1. ✓ MSP 模块正确实例化（515个参数）
2. ✓ MACL 模块正确实例化（49,729个参数）
3. ✓ DHN 采样器正确配置（queue_size=8192, momentum=0.99）
4. ✓ 所有自定义模块的参数都可训练
5. ✓ 数据集路径配置正确
6. ✓ 学习率调度器配置正确（CosineAnnealingLR, T_max=12）
7. ✓ 损失权重合理

---

## 🚀 下一步行动

1. **立即**：修复 Backbone 冻结
2. **立即**：确认或生成 Stage1 预训练权重
3. **训练前**：运行 `python deep_check.py` 再次验证
4. **训练中**：监控 Backbone 参数是否真正冻结
5. **训练后**：检查 MACL 损失是否正常产生

---

生成时间：2025-11-07
检查工具：deep_check.py
