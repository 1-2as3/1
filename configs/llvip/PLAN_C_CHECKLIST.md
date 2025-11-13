# Plan C 执行清单与故障排除指南

## 📋 启动前检查清单

### ✅ 必须完成的步骤
- [ ] 1. 数据配对检查通过 (运行 `python configs/llvip/check_planC_data_pairing.py`)
- [ ] 2. Checkpoint存在: `work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth`
- [ ] 3. GPU显存 ≥ 6GB (batch_size=4需要约5.5GB)
- [ ] 4. 磁盘空间 ≥ 10GB (日志+checkpoint)

### 🎯 关键配置确认
```python
# 必须为True!
train_dataloader.dataset.return_modality_pair = True

# 必须使用双模态预处理器
model['data_preprocessor']['type'] = 'PairedDetDataPreprocessor'

# 保守MACL权重
model['roi_head']['lambda1'] = 0.01  # 不是0.05!

# 温和学习率
optimizer.lr = 5e-5  # 不是1e-5!
```

---

## 🚀 启动训练 (三种方式)

### 方式1: 带监控启动 (推荐)
```bash
python configs/llvip/run_planC_training.py
```
- 自动监控loss_macl和mAP
- 实时警告异常情况

### 方式2: 标准启动
```bash
python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py
```

### 方式3: 后台运行
```bash
python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py > planC.log 2>&1 &
tail -f planC.log  # 监控日志
```

---

## 📊 关键监控指标 (前100 iter)

### 必须出现的Loss项
```
✅ loss_rpn_cls
✅ loss_rpn_bbox
✅ loss_cls
✅ loss_bbox
✅ loss_macl  ← 这是Plan C的核心!如果缺失=配置失败
```

### 正常范围参考
```python
Epoch 1 Iter 50:
    loss_macl:  0.3 ~ 0.6  (初始较高,逐渐下降)
    loss_cls:   0.04 ~ 0.08
    loss_bbox:  0.08 ~ 0.15
    loss_total: 0.15 ~ 0.25  (检测loss) + loss_macl
    grad_norm:  5 ~ 15       (超过20=不稳定)
```

---

## 🎯 成功/失败判定标准

### ✅ 成功信号 (继续训练)
| Epoch | mAP阈值 | loss_macl | 判定 |
|-------|---------|-----------|------|
| 1     | ≥ 0.55  | 收敛至0.2-0.3 | ✅ 正常 |
| 2     | ≥ 0.57  | 稳定在0.15-0.25 | ✅ 继续 |
| 3     | ≥ 0.58  | < 0.2 | ✅ 有望达标 |

### ⚠️ 警告信号 (需调整)
| 情况 | 表现 | 建议调整 |
|------|------|----------|
| Epoch 1 mAP=0.53 | 略低但可救 | lambda1→0.005, lr→3e-5 |
| loss_macl不收敛 | 持续>0.5 | 降低lambda1或检查配对 |
| grad_norm震荡 | >20 | 降低lr或增加warmup |

### 🔴 失败信号 (立即停止)
| 情况 | 判定 | 原因分析 |
|------|------|----------|
| Epoch 1 mAP < 0.52 | 彻底崩溃 | 梯度错向/特征漂移 |
| loss_macl未出现 | 配置错误 | 数据配对失败 |
| loss爆炸 (>10) | 训练失控 | lr过高或lambda1过大 |

---

## 🔧 故障排除手册

### 问题1: loss_macl未出现

**症状**: 训练日志只有loss_cls/loss_bbox,没有loss_macl

**诊断步骤**:
```bash
# 1. 检查配对状态
python configs/llvip/check_planC_data_pairing.py

# 2. 检查配置文件
grep "return_modality_pair" configs/llvip/stage2_2_planC_dualmodality_macl.py
# 应该输出: return_modality_pair=True (2处)

# 3. 检查数据预处理器
grep "PairedDetDataPreprocessor" configs/llvip/stage2_2_planC_dualmodality_macl.py
# 应该输出: type='PairedDetDataPreprocessor'
```

**解决方案**:
- 确认`return_modality_pair=True`
- 确认`data_preprocessor`类型正确
- 清空旧checkpoint重新训练

---

### 问题2: mAP持续低于0.55

**症状**: Epoch 1-2 mAP徘徊在0.52-0.54

**诊断**:
```python
# 查看loss_macl是否过大主导训练
loss_macl / loss_total > 0.5  # 如果True,则MACL过强
```

**解决方案**:
- **降低lambda1**: 0.01 → 0.005
- **提高检测loss权重**: 
  ```python
  model['roi_head']['bbox_head']['loss_cls']['loss_weight'] = 1.5
  ```
- **延长warmup**: 500 → 1000 iter

---

### 问题3: loss_macl不收敛

**症状**: loss_macl持续>0.5,不下降

**可能原因**:
1. 双模态特征差异过大
2. 温度参数tau不合适
3. 投影维度过小

**解决方案**:
```python
# 调整MACL head配置
macl_head=dict(
    tau=0.1,          # 从0.07增加到0.1
    proj_dim=256,     # 从128增加到256
    use_bn=True,      # 确保启用
)
```

---

### 问题4: grad_norm震荡

**症状**: grad_norm在5-25之间剧烈波动

**解决方案**:
```python
# 收紧梯度裁剪
clip_grad=dict(max_norm=5.0, norm_type=2)  # 从10.0降至5.0

# 降低学习率
optimizer=dict(lr=3e-5)  # 从5e-5降至3e-5
```

---

## 📈 进阶调优策略

### 如果Epoch 3 mAP达到0.58+

可以逐步启用更多组件:

```python
# Epoch 3后的配置调整
model['roi_head'].update(dict(
    use_msp=True,              # 启用MSP
    lambda2=0.005,             # 轻量级MSP权重
    use_domain_alignment=False # 暂不启用
))
```

### 如果Epoch 3 mAP达到0.60+

可以激进优化:

```python
# Epoch 5后的配置
model['roi_head'].update(dict(
    lambda1=0.015,             # 增加MACL权重
    lambda2=0.01,              # 增加MSP权重
    use_domain_alignment=True, # 启用域对齐
    lambda3=0.005              # 域对齐权重
))
```

---

## 🎯 最终目标检验

### 训练成功标准
- [ ] Epoch 6 mAP ≥ 0.60
- [ ] loss_macl收敛至 < 0.2
- [ ] grad_norm稳定在 5-10
- [ ] Recall ≥ 0.68

### 如果失败...

**Plan D备选**: 回退到Stage 2.1 Epoch 1
```bash
cp work_dirs/stage2_1_kaist_detonly/epoch_1.pth work_dirs/planD_checkpoint.pth
# 从epoch_1.pth重新开始,更保守的配置
```

**Plan E备选**: 纯检测强化
```bash
# 完全禁用MACL,专注检测性能
use_macl=False
lr=1e-4  # 正常学习率
```

---

## 📞 紧急支持检查点

如果遇到以下情况,立即停止并分析:
1. **GPU OOM**: 降低batch_size至2
2. **训练卡死**: 检查Windows防火墙/杀毒软件
3. **loss=nan**: 降低lr至1e-5,检查数据完整性
4. **显存泄漏**: 设置`persistent_workers=False`

---

## 📝 日志记录建议

每个Epoch结束后记录:
```
Epoch X:
  - mAP: X.XXX
  - loss_macl: X.XXX
  - loss_det: X.XXX
  - grad_norm: X.X
  - 判断: [继续/调整/停止]
  - 备注: [异常现象]
```

这样可以快速回溯问题!
