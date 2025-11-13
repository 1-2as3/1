# 🚀 Linux部署快速修复指南

## 问题原因
Windows和Linux的文件路径不同,导致setup脚本找不到文件。

## 🎯 最简单的解决方案

### 方法1: 使用交互式配置向导 (推荐)

```bash
cd ~/xyz/mmdetection/linux_planC_package

# 添加执行权限
chmod +x *.sh *.py

# 运行配置向导 (自动发现所有路径)
bash configure_wizard.sh
```

**向导会自动:**
- ✅ 搜索KAIST数据集位置
- ✅ 搜索checkpoint文件
- ✅ 让您选择或确认路径
- ✅ 自动更新配置文件
- ✅ 运行双模态烟雾测试
- ✅ 告诉您如何启动训练

**完成后直接运行:**
```bash
bash train_planC.sh
```

---

### 方法2: 手动查找路径

如果向导失败,手动查找:

```bash
cd ~/xyz/mmdetection/linux_planC_package

# 步骤1: 查找KAIST数据集
bash find_paths.sh

# 步骤2: 找到路径后,运行setup
bash setup_planC.sh <数据集路径> <checkpoint路径>

# 例如:
bash setup_planC.sh /mnt/data/KAIST ~/mmdetection/work_dirs/stage1/epoch_48.pth
```

---

### 方法3: 快速诊断

如果不确定问题在哪:

```bash
# 运行环境检查
bash quick_check.sh

# 检查输出中的:
# - Python环境是否正确
# - CUDA是否可用
# - MMDetection是否安装
# - 常见目录是否存在
```

---

## 🔍 常见路径位置

### KAIST数据集可能在:
- `/data/KAIST/`
- `/mnt/data/kaist/`
- `/home/msi-kklt/datasets/KAIST/`
- `/home/msi-kklt/data/kaist_dataset/`

### Checkpoint可能在:
- `~/mmdetection/work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth`
- `~/xyz/mmdetection/work_dirs/stage1/epoch_48.pth`
- `~/checkpoints/stage1_final.pth`

---

## ⚡ 最快路径

如果您已经知道文件位置:

```bash
# 直接运行setup (替换为实际路径)
bash setup_planC.sh /实际的/KAIST/路径 /实际的/checkpoint.pth

# 通过烟雾测试后,启动训练
bash train_planC.sh
```

---

## 📋 检查清单

训练前确认:

```bash
# 1. 数据集存在且结构正确
ls /path/to/kaist/
# 应该看到: visible/  infrared/  annotations/

# 2. Checkpoint存在
ls -lh /path/to/checkpoint.pth
# 应该显示文件大小 (通常>100MB)

# 3. GPU可用
nvidia-smi
# 应该显示GPU状态

# 4. Python环境正确
python -c "import mmdet; print(mmdet.__version__)"
# 应该输出 3.x.x
```

---

## 🆘 仍然失败?

查看详细故障排除文档:
```bash
cat PATH_TROUBLESHOOTING.md
```

或收集诊断信息:
```bash
bash quick_check.sh > diagnostic.log 2>&1
bash find_paths.sh >> diagnostic.log 2>&1
cat diagnostic.log
```

---

## ✅ 成功标志

配置成功后应该看到:

```
========================================
配置完成! 可以开始训练
========================================

启动训练:
  bash train_planC.sh           # 单GPU
  bash train_planC.sh 0,1       # 双GPU

监控训练:
  tail -f work_dirs/planC_*/train_*.log | grep -E 'loss_macl|mAP'
```

然后运行 `bash train_planC.sh` 开始训练!
