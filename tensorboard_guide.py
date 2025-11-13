"""
Plan C TensorBoard 监控完整指南
=================================

功能:
1. 实时监控训练曲线
2. 对比不同实验结果
3. 分析loss组成和趋势

使用方法:
---------

## 步骤1: 启动TensorBoard服务器

打开新的PowerShell窗口,运行:
```powershell
cd C:\Users\Xinyu\mmdetection
tensorboard --logdir=work_dirs --port=6006
```

然后浏览器访问: http://localhost:6006

## 步骤2: 启动训练(在另一个窗口)

```bash
python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py
```

## 步骤3: 监控关键指标

TensorBoard会自动显示以下曲线:

### 📈 Loss曲线 (SCALARS标签)
- `train/loss` - 总loss
- `train/loss_rpn_cls` - RPN分类loss
- `train/loss_rpn_bbox` - RPN回归loss
- `train/loss_cls` - RoI分类loss
- `train/loss_bbox` - RoI回归loss
- `train/loss_macl` - MACL对比学习loss (关键!)
- `train/grad_norm` - 梯度范数

### 📊 验证指标
- `val/pascal_voc/mAP` - 验证集mAP (主指标)
- `val/pascal_voc/AP50` - AP@IoU=0.5
- `val/recall` - 召回率

### 🎯 学习率
- `lr` - 当前学习率
- `momentum` - 动量

---

## 🔍 关键监控点

### Epoch 1 (前100 iter):

#### ✅ 正常信号
```
loss_macl: 0.3 → 0.25 (下降趋势)
loss_cls: 0.05 → 0.04
loss_bbox: 0.10 → 0.09
grad_norm: 8 → 10 (稳定)
```

#### ⚠️ 异常信号
```
loss_macl: 未出现 或 > 1.0
grad_norm: > 20 (震荡)
loss: 出现NaN
```

---

## 🎨 TensorBoard高级功能

### 1. 对比多次实验

在左侧勾选多个run:
- `stage2_1_planB_macl_rescue` (失败基线)
- `stage2_2_planC_dualmodality_macl` (当前训练)

可以看到Plan C是否有改善!

### 2. 平滑曲线

调整Smoothing滑块 (建议0.6) 去除噪声,看清趋势

### 3. 自定义Y轴范围

点击设置图标 → "Fit domain to data" 自动调整

### 4. 下载数据

点击下载按钮导出CSV,可用于绘图

---

## 🚨 常见问题排查

### Q1: TensorBoard显示"No dashboards are active"
**原因**: 训练尚未写入事件文件
**解决**: 等待训练启动,约1-2分钟后刷新

### Q2: loss_macl未出现
**原因**: 双模态配对失败或MACL未启用
**检查**:
```bash
grep "loss_macl" work_dirs/stage2_2_planC_dualmodality_macl/*/vis_data/scalars.json
```

### Q3: 曲线不更新
**原因**: 浏览器缓存
**解决**: Ctrl+F5强制刷新

---

## 📊 Plan C成功判定 (TensorBoard版)

### Epoch 1结束后观察:

#### ✅ 成功 (继续训练)
- mAP曲线: 0.53 → **0.55+**
- loss_macl: 0.4 → **0.2-0.3** (收敛)
- grad_norm: 稳定在 **5-12**

#### ⚠️ 需调整
- mAP: 0.53 → 0.54 (提升缓慢)
- loss_macl: 持续 > 0.4 (不收敛)
- 建议: 降低lambda1或lr

#### 🔴 失败 (立即停止)
- mAP: < 0.52 (下降)
- loss_macl: 未出现或爆炸
- grad_norm: > 20 (失控)

---

## 💡 自动化监控脚本

创建 `monitor_tensorboard.py`:
```python
from tensorboard.backend.event_processing import event_accumulator
import time

def monitor_training():
    ea = event_accumulator.EventAccumulator('work_dirs/stage2_2_planC_dualmodality_macl')
    ea.Reload()
    
    # 获取最新mAP
    if 'val/pascal_voc/mAP' in ea.Tags()['scalars']:
        map_events = ea.Scalars('val/pascal_voc/mAP')
        latest_map = map_events[-1].value
        print(f"Latest mAP: {latest_map:.4f}")
    
    # 获取最新loss_macl
    if 'train/loss_macl' in ea.Tags()['scalars']:
        macl_events = ea.Scalars('train/loss_macl')
        latest_macl = macl_events[-1].value
        print(f"Latest loss_macl: {latest_macl:.4f}")

while True:
    monitor_training()
    time.sleep(60)  # 每分钟检查一次
```

---

## 🎯 实战工作流

### 窗口1: TensorBoard
```bash
tensorboard --logdir=work_dirs --port=6006
# 浏览器打开: http://localhost:6006
```

### 窗口2: 训练
```bash
python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py
```

### 窗口3: 日志监控
```bash
tail -f work_dirs/stage2_2_planC_dualmodality_macl/*/20*.log
# Windows: Get-Content -Wait <log_path>
```

### 浏览器: 实时曲线
- 每5分钟检查一次TensorBoard
- 重点看: loss_macl出现 + mAP趋势

---

## 📸 关键截图时刻

建议在以下时刻截图保存:

1. **Iter 50**: loss_macl首次出现
2. **Iter 500**: warmup结束,loss稳定
3. **Epoch 1**: 第一次验证mAP
4. **Epoch 3**: 中期评估
5. **Epoch 6**: 最终结果

这样可以完整记录训练过程!

---

## 🔗 更多资源

TensorBoard官方文档:
https://www.tensorflow.org/tensorboard/get_started

MMDetection可视化指南:
https://mmdetection.readthedocs.io/en/latest/user_guides/visualization.html
"""

if __name__ == '__main__':
    print(__doc__)
    print("\n" + "="*70)
    print("快速启动命令:")
    print("="*70)
    print("\n1. 启动TensorBoard:")
    print("   tensorboard --logdir=work_dirs --port=6006")
    print("\n2. 打开浏览器:")
    print("   http://localhost:6006")
    print("\n3. 开始训练:")
    print("   python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl.py")
    print("="*70)
