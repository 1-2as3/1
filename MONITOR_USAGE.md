# Training Monitoring Scripts Usage Guide

## 文件清单
1. **monitor_training.sh** - 实时监控脚本（每30秒刷新）
2. **analyze_logs.sh** - 日志分析与 CSV 导出

---

## 1. monitor_training.sh（实时监控）

### 功能
- 实时显示：当前 epoch、mAP、loss、domain_weight、学习率
- GPU 利用率监控
- 训练进度条（12 epochs）
- 自动检测决策点（Epoch 4/6/12）并给出建议
- 最近错误/警告提示

### 使用方法

#### 基础用法（默认 30 秒刷新）
```bash
chmod +x monitor_training.sh
./monitor_training.sh
```

#### 指定工作目录和刷新间隔
```bash
# 监控指定目录，每 15 秒刷新一次
./monitor_training.sh work_dirs/stage2_kaist_full_conservative_remote 15
```

#### 后台运行（配合 tmux/screen）
```bash
# 创建新的 tmux 会话
tmux new -s monitor

# 在 tmux 内运行监控
./monitor_training.sh

# 分离会话：按 Ctrl+B 然后按 D
# 重新连接：tmux attach -t monitor
```

### 输出示例
```
========================================
   MMDetection Stage2 Training Monitor
========================================
Work Dir: work_dirs/stage2_kaist_full_conservative_remote
Refresh: 30s  |  Press Ctrl+C to exit
========================================

Log File: 20251110_143022.log

=== GPU Status ===
  GPU 0 (NVIDIA GeForce RTX 4090): 85% GPU, 18432/24576 MB

=== Training Progress ===
  Epoch: 6/12  [██████████░░░░░░░░░░] 50%

Current Epoch: 6

=== Validation Metrics ===
  mAP:         0.5823
  AP50:        0.6154

=== Training Losses ===
  Total Loss:  0.3421
  Loss Cls:    0.1234
  Loss BBox:   0.2187

=== Hyperparameters ===
  Domain Weight: 0.0600
  Learning Rate: 2.3e-04

=== Decision Points ===
⚠ Epoch 6: mAP < 0.60 - Consider switching to Plan C
  Command: python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \
           --work-dir work_dirs/stage2_kaist_full_C \
           --cfg-options custom_hooks.0.target_domain_weight=0.06 \
                         custom_hooks.0.warmup_epochs=6

Last updated: 2025-11-10 14:32:15
```

---

## 2. analyze_logs.sh（日志分析）

### 功能
- 输出每个 epoch 的 mAP 与 domain_weight 表格
- 统计错误/警告数量
- 生成 CSV 文件用于外部绘图（Excel/Python/gnuplot）
- 显示最佳 checkpoint 信息

### 使用方法

#### 基础用法
```bash
chmod +x analyze_logs.sh
./analyze_logs.sh work_dirs/stage2_kaist_full_conservative_remote
```

#### 输出示例
```
Analyzing: work_dirs/stage2_kaist_full_conservative_remote/20251110_143022.log

=== Epoch Summary ===
Epoch  1: mAP=0.6285  domain_weight=0.0500
Epoch  2: mAP=0.6012  domain_weight=0.0625
Epoch  3: mAP=0.5501  domain_weight=0.0750
Epoch  4: mAP=0.5734  domain_weight=0.0625
Epoch  5: mAP=0.5891  domain_weight=0.0688
Epoch  6: mAP=0.5823  domain_weight=0.0600

=== Training Loss Trend (Last 10 epochs) ===
Epoch  1: loss=0.5234
Epoch  2: loss=0.4512
Epoch  3: loss=0.4021
Epoch  4: loss=0.3892
Epoch  5: loss=0.3567
Epoch  6: loss=0.3421

=== Best Checkpoint ===
Latest checkpoint: epoch_6.pth
Best model: best_pascal_voc_mAP_epoch_1.pth (512M)

=== Error Summary ===
Total errors: 0
Total warnings: 3

=== Recent Issues (Last 5) ===
[WARNING] Low AP for small objects: consider increasing input scale
[WARNING] Domain loss gradient vanishing at epoch 4
[WARNING] RPN proposals < 1000 in 5% of images

=== Generate CSV for external plotting ===
CSV saved to: work_dirs/stage2_kaist_full_conservative_remote/metrics_20251110_143522.csv
You can plot this with: gnuplot -e "set datafile separator ','; plot 'metrics.csv' using 1:2 with lines title 'mAP'"
```

---

## 3. 组合使用（推荐工作流）

### 场景 A：启动训练后立即监控
```bash
# Terminal 1: 启动训练
python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \
    --work-dir work_dirs/stage2_kaist_full_conservative_remote

# Terminal 2: 实时监控（另开一个窗口）
./monitor_training.sh work_dirs/stage2_kaist_full_conservative_remote 30
```

### 场景 B：使用 tmux 多窗口管理
```bash
# 创建 tmux 会话并分屏
tmux new -s training

# 在 tmux 内分屏（Ctrl+B 然后按 %）
# 左侧：训练进程
python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \
    --work-dir work_dirs/stage2_kaist_full_conservative_remote

# 右侧：切换到右窗格（Ctrl+B 然后按方向键）
./monitor_training.sh work_dirs/stage2_kaist_full_conservative_remote

# 分离 tmux：Ctrl+B 然后按 D
# 重新连接：tmux attach -t training
```

### 场景 C：训练完成后快速分析
```bash
# 查看完整 epoch 总结与导出 CSV
./analyze_logs.sh work_dirs/stage2_kaist_full_conservative_remote

# 用 Python 绘制曲线（需要 matplotlib）
python <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('work_dirs/stage2_kaist_full_conservative_remote/metrics_*.csv')
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['mAP'], marker='o', label='mAP')
plt.plot(df['epoch'], df['domain_weight'], marker='s', label='Domain Weight')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png')
print('Saved to training_curves.png')
PY
```

---

## 4. 常见问题

### Q1: monitor_training.sh 提示 "bc: command not found"
```bash
# Ubuntu/Debian
sudo apt-get install -y bc

# CentOS/RHEL
sudo yum install -y bc
```

### Q2: 监控脚本无法找到日志文件
确认 `work_dir` 路径正确且训练已开始：
```bash
ls -lh work_dirs/stage2_kaist_full_conservative_remote/*.log
```

### Q3: 如何在 Windows 本地监控远程训练？
使用 SSH 转发日志：
```powershell
# PowerShell 实时拉取日志
ssh user@remote "tail -f ~/mmdetection_remote/work_dirs/*/20*.log"
```

### Q4: 如何设置自动邮件提醒（Epoch 6 决策点）？
在 `monitor_training.sh` 的 `check_decision_points` 函数中添加：
```bash
if [ "$epoch" -eq 6 ] && (( $(echo "$map < 0.60" | bc -l) )); then
    echo "Epoch 6 alert: mAP < 0.60" | mail -s "Training Alert" your@email.com
fi
```

---

## 5. 脚本自定义

### 调整刷新频率
编辑 `monitor_training.sh` 第 5 行：
```bash
REFRESH_SEC="${2:-30}"  # 改为 10 / 60 等
```

### 添加自定义指标
在 `extract_metrics` 函数中添加：
```bash
local custom_metric=$(grep -oP 'your_metric:\s+\K[\d.]+' "$log_file" | tail -n1)
echo -e "  Custom Metric: ${custom_metric:-N/A}"
```

### 修改决策点阈值
在 `check_decision_points` 函数中调整：
```bash
if (( $(echo "$map < 0.55" | bc -l) )); then  # 从 0.60 改为 0.55
```

---

## 6. 快速命令参考卡

| 任务 | 命令 |
|------|------|
| 启动实时监控 | `./monitor_training.sh` |
| 指定刷新间隔（15秒） | `./monitor_training.sh work_dirs/xxx 15` |
| 后台运行监控 | `nohup ./monitor_training.sh > monitor.out 2>&1 &` |
| 停止监控 | `Ctrl+C` 或 `pkill -f monitor_training.sh` |
| 分析日志生成 CSV | `./analyze_logs.sh work_dirs/xxx` |
| 查看最近 50 行日志 | `tail -n 50 work_dirs/*/20*.log` |
| 搜索错误 | `grep -i error work_dirs/*/20*.log` |
| 实时跟踪日志 | `tail -f work_dirs/*/20*.log` |

---

## 7. 上传到远程服务器

```bash
# 从本地上传脚本到远程
scp monitor_training.sh analyze_logs.sh user@remote:~/mmdetection_remote/

# 远程执行
ssh user@remote
cd mmdetection_remote
chmod +x monitor_training.sh analyze_logs.sh
./monitor_training.sh
```

---

完成！现在你拥有完整的训练监控工具链。
