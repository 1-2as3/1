# Plan C 问题诊断与解决方案

## 问题总结

### 问题1: 训练卡死
**症状**: 日志停在"Checkpoints will be saved..."之后
**原因**: Windows系统下双模态数据加载(return_modality_pair=True)与多进程DataLoader死锁
**解决**: 
- `num_workers: 2 → 0`
- `persistent_workers: True → False`
- `batch_size: 4 → 2`

### 问题2: UnicodeEncodeError
**症状**: `UnicodeEncodeError: 'gbk' codec can't encode character '\u2713'`
**原因**: Windows控制台GBK编码与Python UTF-8字符冲突
**解决**: 
- 创建无Unicode字符的clean配置文件
- 启动脚本中设置`chcp 65001`(UTF-8)

## 已修复的配置

### 新配置文件
`configs/llvip/stage2_2_planC_dualmodality_macl_clean.py`
- 移除所有Unicode特殊字符(✓等)
- 使用纯ASCII英文注释
- 核心配置不变

### 关键修复项

```python
# 1. DataLoader配置
train_dataloader = dict(
    batch_size=2,               # 降低batch
    num_workers=0,              # 单进程加载
    persistent_workers=False,   # 非持久化
)

# 2. 数据预处理器
model['data_preprocessor'] = dict(
    type='PairedDetDataPreprocessor',  # 必须使用配对预处理器
    ...
)

# 3. 双模态启用
dataset=dict(
    return_modality_pair=True,  # 启用双模态配对
    ...
)
```

## 启动训练 (3种方式)

### 方式1: 推荐启动脚本
```bash
start_planC_final.bat
```
- 自动设置UTF-8编码
- 使用正确的Python解释器
- 使用clean配置文件

### 方式2: 手动启动
```bash
chcp 65001
C:\Users\Xinyu\.conda\envs\py311\python.exe tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl_clean.py
```

### 方式3: TensorBoard + 训练
```bash
# 窗口1: TensorBoard
tensorboard --logdir=work_dirs --port=6006

# 窗口2: 训练
start_planC_final.bat
```

## 预期行为

### 正常启动标志 (前5分钟)

```log
2025/11/12 XX:XX:XX - mmengine - INFO - Load checkpoint from ...
2025/11/12 XX:XX:XX - mmengine - INFO - Checkpoints will be saved to ...
2025/11/12 XX:XX:XX - mmengine - INFO - Epoch(train) [1][  50/22878]
    loss: 0.3xxx
    loss_macl: 0.4xxx  ← 必须出现!
    loss_cls: 0.0xxx
    loss_bbox: 0.1xxx
    grad_norm: 9.xxx
```

### 异常信号

| 现象 | 原因 | 解决 |
|------|------|------|
| 卡在"Checkpoints will be saved" | DataLoader死锁 | 确认num_workers=0 |
| UnicodeEncodeError | 控制台编码 | 使用start_planC_final.bat |
| loss_macl未出现 | 配对失败 | 检查return_modality_pair=True |
| OOM错误 | 显存不足 | 降低batch_size至1 |

## Smoke Test 结果

已通过的测试:
- [✓] MMDetection模块导入
- [✓] 配置文件加载
- [✓] 数据路径检查
- [✓] Checkpoint存在性检查

待实际训练验证:
- [ ] 数据加载不卡死
- [ ] loss_macl正常出现
- [ ] mAP回升至0.55+

## TensorBoard监控

启动后访问: http://localhost:6006

关键指标:
- `train/loss_macl` - 应从0.4降至0.2
- `train/grad_norm` - 应在5-12之间
- `val/pascal_voc/mAP` - Epoch 1应≥0.55

## 故障排除快速参考

### 如果还是卡住:
```bash
# 1. 停止所有Python进程
taskkill /F /IM python.exe

# 2. 删除旧运行目录
rd /s /q work_dirs\stage2_2_planC_dualmodality_macl\20251112_*

# 3. 确认配置
grep -n "num_workers" configs/llvip/stage2_2_planC_dualmodality_macl_clean.py

# 4. 重新启动
start_planC_final.bat
```

### 如果看到UnicodeError:
```bash
# 确保使用clean配置和final启动脚本
start_planC_final.bat
```

### 如果loss_macl未出现:
```bash
# 检查数据配对
python configs/llvip/check_planC_data_pairing.py

# 检查配置
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/llvip/stage2_2_planC_dualmodality_macl_clean.py')
print('return_modality_pair:', cfg.train_dataloader.dataset.return_modality_pair)
print('data_preprocessor:', cfg.model.data_preprocessor.type)
print('use_macl:', cfg.model.roi_head.use_macl)
"
```

## 下一步

1. **立即启动**: 运行 `start_planC_final.bat`
2. **监控前5分钟**: 确认loss_macl出现
3. **TensorBoard**: 在 http://localhost:6006 查看曲线
4. **Epoch 1判定**: 80分钟后检查mAP

## 文件清单

已创建的文件:
- `configs/llvip/stage2_2_planC_dualmodality_macl_clean.py` - 干净配置
- `start_planC_final.bat` - 最终启动脚本
- `test_planC_smoke.py` - Smoke test脚本
- `tensorboard_guide.py` - TensorBoard使用指南
- `PLANC_FIX_README.md` - 本文档

旧文件(可删除):
- `configs/llvip/stage2_2_planC_dualmodality_macl.py` - 有Unicode字符
- `start_planC_with_tensorboard.bat` - 旧启动脚本
- `start_planC_safe.bat` - 中间版本

## 成功标准

### Epoch 1 (80分钟):
- mAP ≥ 0.55 ✓
- loss_macl < 0.3 ✓
- grad_norm 5-12 ✓

### Epoch 3 (4小时):
- mAP ≥ 0.58 ✓
- loss_macl < 0.2 ✓

### Epoch 6 (8小时):
- mAP ≥ 0.60 ✓ (目标!)

Good luck! 🚀
