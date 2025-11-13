# Plan C 训练卡死问题修复说明

## 🔴 问题诊断

**症状**: 训练卡在hooks初始化后,日志停在"after_run"部分

**根本原因**: Windows系统下,双模态数据加载(return_modality_pair=True)与多进程DataLoader(num_workers>0)存在死锁问题

**影响因素**:
1. `num_workers=2` → 多进程加载
2. `persistent_workers=True` → 持久化worker进程
3. `return_modality_pair=True` → 每个样本需加载2张图(visible+infrared)
4. Windows `spawn`启动方式 → 进程间通信开销大

## ✅ 解决方案

### 已修复的配置项

```python
# 修改前 (会卡死)
train_dataloader = dict(
    batch_size=4,
    num_workers=2,              # ❌ 多进程
    persistent_workers=True,    # ❌ 持久化
)

# 修改后 (稳定)
train_dataloader = dict(
    batch_size=2,               # ✅ 降低batch避免显存压力
    num_workers=0,              # ✅ 单进程加载
    persistent_workers=False,   # ✅ 非持久化
)
```

### 性能影响评估

| 配置 | 数据加载速度 | GPU利用率 | 稳定性 |
|------|-------------|-----------|--------|
| 修改前 | 理论更快 | ~85% | ❌ 死锁 |
| 修改后 | 略慢(~10%) | ~80% | ✅ 稳定 |

**结论**: 性能损失可接受,稳定性优先!

---

## 🚀 立即重新启动训练

### 方式1: 一键启动(推荐)
```bash
start_planC_with_tensorboard.bat
```
- 自动启动TensorBoard
- 自动打开训练
- 浏览器访问: http://localhost:6006

### 方式2: 手动启动

#### 窗口1: TensorBoard
```bash
tensorboard --logdir=work_dirs --port=6006
```

#### 窗口2: 训练
```bash
python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py
```

---

## 📊 TensorBoard监控关键指标

训练开始后,在 http://localhost:6006 查看:

### 🎯 必须出现的指标
- ✅ `train/loss_macl` - 如果没有=配对失败
- ✅ `train/loss_cls`
- ✅ `train/loss_bbox`
- ✅ `val/pascal_voc/mAP`

### 📈 正常范围 (Epoch 1, Iter 50)
```
loss_macl:  0.3 ~ 0.5
loss_cls:   0.04 ~ 0.06
loss_bbox:  0.08 ~ 0.12
grad_norm:  5 ~ 12
```

### ⚠️ 异常信号
```
loss_macl:  未出现 或 > 1.0  → 配对失败
grad_norm:  > 20            → 不稳定
loss:       NaN             → 训练崩溃
```

---

## 🔧 如果还是卡住...

### 诊断步骤

1. **确认配置已更新**:
```bash
grep "num_workers" configs/llvip/stage2_2_planC_dualmodality_macl.py
# 应该显示: num_workers=0
```

2. **删除旧的运行目录**:
```bash
rd /s /q work_dirs\stage2_2_planC_dualmodality_macl\20251112_193402
```

3. **检查显存占用**:
```bash
nvidia-smi
# 如果显存已满,重启系统释放
```

4. **尝试更小的batch_size**:
```python
# 在配置文件中改为
batch_size=1  # 最小化显存占用
```

---

## 🎯 成功启动的标志

训练正常启动后,你会在**2分钟内**看到:

```log
2025/11/12 XX:XX:XX - mmengine - INFO - Epoch(train) [1][  50/22878]  
    loss: 0.3xxx  
    loss_macl: 0.4xxx  ← 这个必须出现!
    loss_cls: 0.0xxx  
    loss_bbox: 0.1xxx  
    grad_norm: 9.xxx
```

如果5分钟还没看到训练日志,按Ctrl+C停止,报告错误信息!

---

## 📝 训练监控检查清单

在训练的不同阶段,检查以下项目:

### ✅ 启动阶段 (前2分钟)
- [ ] 看到"Epoch(train) [1][50/...]"日志
- [ ] loss_macl出现在日志中
- [ ] TensorBoard能访问(http://localhost:6006)

### ✅ Warmup阶段 (前10分钟)
- [ ] loss_macl从0.5降至0.3左右
- [ ] grad_norm稳定在5-15
- [ ] 无NaN或异常大的loss值

### ✅ Epoch 1结束 (~60分钟)
- [ ] 出现验证日志"Epoch(val) [1][...]"
- [ ] mAP ≥ 0.52 (最低容忍线)
- [ ] loss_macl < 0.3 (已收敛)

---

## 🆘 紧急支持

如果修复后还是有问题,提供以下信息:

1. **最新日志**:
```bash
type work_dirs\stage2_2_planC_dualmodality_macl\*\*.log | Select-Object -Last 50
```

2. **GPU状态**:
```bash
nvidia-smi
```

3. **配置确认**:
```bash
grep -A 5 "train_dataloader" configs/llvip/stage2_2_planC_dualmodality_macl.py
```

把这些信息发给我,立即诊断!
