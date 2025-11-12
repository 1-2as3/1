# Stage2.1 Recovery Training - Monitoring Guide

## 当前训练状态
**开始时间**: 2025/11/11 18:06:00
**配置文件**: `configs/llvip/stage2_1_kaist_detonly_pure_detection.py`
**工作目录**: `work_dirs/stage2_1_pure_detection/20251111_180600`
**训练策略**: Plan A - Pure Detection (无MACL/DHN/Domain损失)

## 监控工具

### 1. 实时批处理监控 (每30秒刷新)
```bash
monitor_plan_a_live.bat
```
显示:
- 当前训练进度 (epoch, batch)
- 最近10个batch的loss趋势
- 验证结果
- GPU状态
- Checkpoint文件列表

### 2. 详细Python分析
```bash
python monitor_training_detailed.py
```
显示:
- 当前进度和loss详情
- Loss趋势分析 (最近100个batch的平均值和趋势)
- 所有验证结果历史
- 与baseline对比

### 3. 通用监控 (支持Plan A和Plan B)
```bash
monitor_recovery_training.bat
```
显示:
- Plan A和Plan B的状态
- GPU内存和温度
- Early Stop警告
- 所有工作目录的检查点

### 4. 手动查看最新日志
```bash
Get-Content "work_dirs\stage2_1_pure_detection\20251111_180600\20251111_180600.log" -Tail 20
```

## 关键指标监控

### 训练阶段 (Epoch 1-5)
| 指标 | 健康范围 | 当前状态 | 说明 |
|------|----------|----------|------|
| loss_total | 0.25-0.35 | ✅ 0.28 | 总损失,应该逐渐下降 |
| loss_cls | 0.10-0.15 | ✅ 0.10 | 分类损失 |
| loss_bbox | 0.15-0.20 | ✅ 0.16 | 边界框回归损失 |
| acc | >90% | ✅ 96.88% | 分类准确率 |
| grad_norm | <10 | ✅ ~7 | 梯度范数 (有clip_grad=3.0保护) |

### 验证阶段 (每个epoch结束)
| Epoch | 预期mAP | 状态判断 |
|-------|---------|----------|
| 1 | ≥0.60 | 高于失败baseline (0.5265) |
| 2 | ≥0.61 | 稳定上升 |
| 3 | ≥0.62 | 接近目标 |
| 4 | ≥0.63 | **达到目标** |
| 5 | ≥0.63 | 确认稳定 |

## 训练时间估算
- 每个epoch: ~4小时10分钟
- 总训练时间 (5 epochs): ~20-22小时
- 预计完成时间: 2025/11/12 14:00-16:00

## 紧急情况处理

### 🟢 正常情况 (当前)
- Loss稳定下降
- Accuracy >90%
- GPU温度 <80°C
- **Action**: 继续监控,无需干预

### 🟡 Loss波动
- Loss短期内上升 >10%
- **Action**: 等待100个batch观察趋势,可能是数据波动

### 🔴 训练崩溃
- Loss爆炸 (>1.0)
- GPU OOM
- 进程意外退出
- **Action**: 
  1. 检查日志文件最后50行
  2. 使用最近的checkpoint (epoch_N.pth) 继续训练
  3. 降低batch_size从4到2

### 🔴 mAP不达标
如果Epoch 5结束时 mAP < 0.63:
1. **分析原因**:
   ```bash
   python monitor_training_detailed.py
   ```
   查看loss趋势是否持续下降

2. **Plan B策略**:
   启动progressive MACL warmup:
   ```bash
   python tools/train.py configs/llvip/stage2_1_kaist_detonly_progressive_macl.py
   ```

## 成功标准

### ✅ Plan A成功
- 最终mAP ≥ 0.63
- Recall ≥ 0.80
- Loss_total稳定在0.25-0.30
- **结论**: Gradient conflict假设证实,MACL确实干扰检测

### ⚠️ Plan A部分成功
- 最终mAP 0.60-0.63
- **Action**: 继续训练3个epoch或启动Plan B

### ❌ Plan A失败
- 最终mAP < 0.60
- **Action**: 必须启动Plan B测试progressive warmup

## 下一步计划

### Plan A成功后
1. 保存最佳checkpoint作为 `stage2_1_recovered.pth`
2. 准备Stage2.2配置:
   - 基于recovered checkpoint
   - 重新引入MACL但降低λ1到0.01
   - 添加DHN和Domain Alignment
3. 更新文档和分析报告

### Plan B (如需要)
- Lambda1 warmup: 0.0 → 0.01 (3 epochs)
- 验证最小干扰阈值
- 对比Plan A和Plan B的最终性能

## 实验记录
- [x] Stage1: LLVIP pretraining (epoch_21, mAP=0.6288)
- [x] Stage2.1 original: KAIST fine-tuning (FAILED at epoch_3, mAP=0.5265)
- [ ] Stage2.1 Plan A: Pure detection recovery (IN PROGRESS)
  - Started: 2025/11/11 18:06
  - Status: Epoch 1, batch 2250/11439 (19.7%)
  - Loss: 0.2607 (healthy)
- [ ] Stage2.1 Plan B: Progressive MACL (PENDING)
- [ ] Stage2.2: Full contrastive+domain training (PENDING)

## 早停触发与恢复策略

### 触发日志关键片段
```
Epoch 2: mAP=0.5884 (Reset counter)
Epoch 3: mAP=0.5542 (<0.58, 1/2)
Epoch 4: mAP=0.5276 (<0.58, 2/2 → trigger)
```

### 误触发原因分析
- 余弦退火使 lr 过早下降到 ~2.5e-5，恢复弹性不足
- KAIST 验证集存在短期结构波动（dets/recall 同步下降）
- EarlyStop 阈值 0.58 与当前恢复阶段的波动区间接近，易误判

### 恢复执行顺序（推荐）
1. 备份 Epoch2 最优权重：
   ```bat
   copy work_dirs\stage2_1_pure_detection\best_pascal_voc_mAP_epoch_2.pth work_dirs\stage2_1_pure_detection\stage2_1_backup_ep2.pth
   ```
2. 启动恢复训练（常数 lr=4e-5，禁 EarlyStop，继续到 8 epochs）：
   ```bat
   C:\Users\Xinyu\.conda\envs\py311\python.exe tools\train.py configs\llvip\stage2_1_kaist_detonly_pure_detection_resume_ep2.py
   ```
3. 监控趋势：连续两轮 mAP ≥0.60 或单次 ≥0.63 即判定成功
4. 成功后：保存最佳权重为 `stage2_1_recovered.pth` 作为 Stage2.2 起点

### 判定条件表
| 状态 | 标准 | 行动 |
|------|------|------|
| 成功 | 单次 mAP ≥0.63 | 进入 Stage2.2 准备 MACL 微弱引入 |
| 稳定 | 连续 ≥0.60 且无下降趋势 | 再补 1 epoch 确认后进入 Stage2.2 |
| 停滞 | 3 个 epoch 都在 0.55~0.59 | 轻调 lr 或增 batch_size |
| 退化 | 连续下降且 loss 不再降 | 引入 Plan B (λ1 warmup 0→0.01) |

### 可选微调策略（若恢复不理想）
- lr 微升到 4.5e-5（不使用调度）
- batch_size=6（显存允许时）
- 打开 fp16（轻微加速，不直接影响 mAP）
- 延迟再引入 EarlyStop（阈值降到 0.55，patience=3，begin=5）

### Plan B 启动条件（严格）
- 恢复训练满 3 个额外 epoch（共 ≥5）仍 <0.60 且趋势平缓
- 或出现 mAP 与 recall 同步持续 2 次下降（结构性退化）

### 后续 Stage2.2 注意事项
- 初始 λ1 设为 0.005 而非 0.05
- 使用分段 warmup: 0→0.005 (2 epochs)，再视稳定性增至 0.01
- 只在 bbox/cls loss 稳定（方差缩小）后再考虑引入 DHN / DomainAlignment

