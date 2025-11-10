# 配对模态（可见光+红外）实现总结

## 📋 概述

成功实现了 LLVIP 数据集的配对模态训练流程，使模型能够同时接收可见光和红外图像，并通过 MACL（Modal-Aware Contrastive Learning）进行跨模态表征学习。

## ✅ 实现的功能

### 1. 数据加载层 (`LLVIPDataset`)

**文件**: `mmdet/datasets/llvip_dataset.py`

**核心改动**:
- 新增 `return_modality_pair` 参数控制是否返回配对输入
- 当启用配对模式时，`__getitem__` 返回格式为:
  ```python
  {
      'inputs': {
          'visible': vis_tensor,    # 可见光图像
          'infrared': ir_tensor     # 红外图像
      },
      'data_samples': DetDataSample
  }
  ```
- 自动构建配对路径（visible ↔ infrared）
- 在 metainfo 中记录配对状态和路径

**配置使用**:
```python
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='LLVIPDataset',
        return_modality_pair=True  # ✅ 启用配对模态
    )
)
```

### 2. 特征提取层 (`TwoStageDetector.extract_feat`)

**文件**: `mmdet/models/detectors/two_stage.py`

**核心改动**:
- 支持 dict 类型输入（包含 'visible' 和 'infrared' 键）
- 分别通过 backbone 和 neck 处理两个模态
- 返回配对特征:
  ```python
  {
      'vis': vis_feats,  # 可见光 FPN 特征 (tuple of 5 levels)
      'ir': ir_feats     # 红外 FPN 特征 (tuple of 5 levels)
  }
  ```

**代码逻辑**:
```python
if isinstance(batch_inputs, dict) and 'visible' in batch_inputs:
    vis_x = self.backbone(batch_inputs['visible'])
    ir_x = self.backbone(batch_inputs['infrared'])
    if self.with_neck:
        vis_x = self.neck(vis_x)
        ir_x = self.neck(ir_x)
    return {'vis': vis_x, 'ir': ir_x}
```

### 3. 损失计算层 (`TwoStageDetector.loss`)

**文件**: `mmdet/models/detectors/two_stage.py`

**核心改动**:
- 检测配对特征格式
- RPN 使用可见光特征进行区域提议
- 特征传递给 RoI Head 进行后续处理

**代码逻辑**:
```python
x = self.extract_feat(batch_inputs)
x_for_rpn = x['vis'] if isinstance(x, dict) and 'vis' in x else x
rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
    x_for_rpn, rpn_data_samples, proposal_cfg=proposal_cfg)
```

### 4. RoI Head 损失计算 (`StandardRoIHead.loss`)

**文件**: `mmdet/models/roi_heads/standard_roi_head.py`

**核心改动**:
- 检测配对模态特征（dict with 'vis' and 'ir' keys）
- 使用可见光特征进行标准检测流程（bbox, mask）
- 对配对特征计算 MACL 对比损失

**MACL 损失计算流程**:
```python
if is_paired_modality and use_macl:
    # 1. 对每个 FPN 层进行全局平均池化
    for vis_fm, ir_fm in zip(vis_feats, ir_feats):
        vis_pooled = F.adaptive_avg_pool2d(vis_fm, 1).flatten(1)
        ir_pooled = F.adaptive_avg_pool2d(ir_fm, 1).flatten(1)
        vis_pooled_list.append(vis_pooled)
        ir_pooled_list.append(ir_pooled)
    
    # 2. 拼接所有层的特征
    vis_feat_vec = torch.cat(vis_pooled_list, dim=1)  # (B, 256*5=1280)
    ir_feat_vec = torch.cat(ir_pooled_list, dim=1)    # (B, 256*5=1280)
    
    # 3. 调用 MACL head 计算对比损失
    macl_out = self.macl_head(vis_feat_vec, ir_feat_vec)
    losses['loss_macl'] = macl_out['loss_macl']
```

### 5. MACL Head 修正

**文件**: `mmdet/models/macldhnmsp/macl_head.py`

**核心改动**:
- 修正 `in_dim` 默认值为 **1280** (256 × 5 FPN levels)
- 投影网络: 1280 → 512 → 256 → 128
- InfoNCE 对比损失 + 可选的 DHN 困难负样本挖掘

**网络结构**:
```python
self.proj = nn.Sequential(
    nn.Linear(1280, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Linear(256, proj_dim)  # proj_dim=128
)
```

## 🧪 测试结果

### 测试脚本: `test_paired_modality.py`

**测试流程**:
1. ✅ 构建模型（use_macl=True, use_msp=True）
2. ✅ 创建配对输入（visible + infrared）
3. ✅ 特征提取（返回 dict with 'vis' and 'ir'）
4. ✅ 损失计算（包含 loss_macl）
5. ✅ 反向传播（MACL 模块有 9 个参数接收梯度）

**测试输出**:
```
[OK] 成功提取配对模态特征
    - 可见光特征层数: 5
    - 红外特征层数: 5
    - 可见光 P2 shape: torch.Size([2, 256, 56, 56])
    - 红外 P2 shape: torch.Size([2, 256, 56, 56])

[OK] 损失计算成功
    损失项:
      - loss_rpn_cls: 0.7369 (list, 5 items)
      - loss_rpn_bbox: 0.0191 (list, 5 items)
      - loss_cls: 0.8124
      - loss_bbox: 0.0349
      - loss_macl: 1.4009  ✅ MACL 损失成功计算

[OK] 反向传播完成
    - 有梯度的参数数量: 170
    - MACL 模块有梯度的参数: 9  ✅ MACL 参数接收梯度
```

## 📊 训练配置

### Stage 1: LLVIP 预训练配置

**文件**: `configs/llvip/stage1_llvip_pretrain.py`

```python
model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        use_macl=True,   # ✅ 启用 MACL 跨模态对比学习
        use_msp=True,    # ✅ 启用 MSP 多尺度模式重加权
        use_dhn=False,   # ❌ 阶段一不使用 DHN
    )
)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='LLVIPDataset',
        img_prefix='C:/LLVIP/LLVIP/',
        ann_file='C:/LLVIP/LLVIP/Annotations',
        return_modality_pair=True  # ✅ 返回配对输入
    )
)
```

## 🔄 数据流图

```
LLVIP Dataset (return_modality_pair=True)
    │
    ├─ visible image  ────┐
    └─ infrared image ────┤
                          │
                          ▼
              {'visible': vis_tensor, 'infrared': ir_tensor}
                          │
                          ▼
              TwoStageDetector.extract_feat()
                          │
          ┌───────────────┴───────────────┐
          │                               │
    backbone(vis)                   backbone(ir)
          │                               │
      neck(vis)                       neck(ir)
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
              {'vis': vis_feats, 'ir': ir_feats}
                          │
                          ▼
              StandardRoIHead.loss()
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    RPN(vis_feats)   Detection(vis_feats)  MACL(vis_feats, ir_feats)
          │               │                       │
    loss_rpn_*      loss_cls/bbox           loss_macl
          │               │                       │
          └───────────────┴───────────────────────┘
                          │
                          ▼
                  Total Loss (聚合所有损失)
```

## 🎯 核心设计理念

### 1. **模块化设计**
- 通过 `return_modality_pair` 开关控制数据模式
- 通过 `use_macl` 开关控制损失计算
- 向后兼容：关闭开关时退化为标准单模态训练

### 2. **共享权重策略**
- 可见光和红外图像共享同一个 backbone 和 neck
- 这种设计促使网络学习模态不变的特征表示
- MACL 损失进一步对齐两个模态的特征空间

### 3. **渐进式训练**
- **Stage 1 (LLVIP)**: 学习跨模态表示（MACL + MSP）
- **Stage 2 (KAIST)**: 域适应（Domain Alignment）
- **Stage 3 (Joint)**: 联合微调（DHN + 全部模块）

## 🔧 关键参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `return_modality_pair` | LLVIPDataset | False | 是否返回配对输入 |
| `use_macl` | StandardRoIHead | False | 是否启用 MACL 损失 |
| `in_dim` | MACLHead | 1280 | 输入特征维度（5×256 FPN） |
| `proj_dim` | MACLHead | 128 | 投影空间维度 |
| `tau` | MACLHead | 0.07 | 对比学习温度参数 |

## 🚀 后续工作

1. **数据集验证**: 在真实 LLVIP 数据集上验证配对加载
2. **超参数调优**: 调整 tau、学习率、batch size
3. **可视化**: 添加特征空间 t-SNE 可视化
4. **性能评估**: 对比单模态 vs 配对模态的检测性能
5. **Stage 2 实现**: KAIST 数据集的域适应训练

## 📝 相关文件

- `mmdet/datasets/llvip_dataset.py` - 数据加载
- `mmdet/models/detectors/two_stage.py` - 特征提取
- `mmdet/models/roi_heads/standard_roi_head.py` - 损失计算
- `mmdet/models/macldhnmsp/macl_head.py` - MACL 对比学习
- `configs/llvip/stage1_llvip_pretrain.py` - 训练配置
- `test_paired_modality.py` - 功能测试

---

**实现日期**: 2025年11月3日  
**状态**: ✅ 完成并测试通过
