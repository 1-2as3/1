# 🎯 Linux部署包已更新 - 下一步操作

## 📦 更新内容

已添加5个新文件来解决路径问题:

1. **configure_wizard.sh** ⭐ - 交互式自动配置工具 (最推荐!)
2. **find_paths.sh** - 自动搜索KAIST数据集和checkpoint
3. **quick_check.sh** - 环境诊断工具
4. **QUICK_FIX.md** - 路径问题快速修复指南
5. **PATH_TROUBLESHOOTING.md** - 详细故障排除文档

## 🚀 立即操作步骤

### 步骤1: 将新文件上传到Linux服务器

**方法A: 使用Git (推荐)**
```bash
# 在Windows上:
cd C:\Users\Xinyu\mmdetection
git add linux_planC_package/*.sh linux_planC_package/*.md
git commit -m "Add path discovery and auto-configuration tools"
git push

# 在Linux服务器上:
cd ~/xyz/mmdetection
git pull
```

**方法B: 使用向日葵远程桌面**
1. 连接到Linux服务器
2. 将 `linux_planC_package/` 整个文件夹拖拽传输
3. 覆盖原有文件

**方法C: 使用SCP**
```bash
# 在Windows PowerShell上:
scp linux_planC_package/*.sh linux_planC_package/*.md msi-kklt@server:~/xyz/mmdetection/linux_planC_package/
```

### 步骤2: 在Linux服务器上运行配置向导

```bash
# SSH连接或通过向日葵打开终端
cd ~/xyz/mmdetection/linux_planC_package

# 添加执行权限
chmod +x *.sh *.py

# 运行自动配置向导
bash configure_wizard.sh
```

**配置向导会:**
- ✅ 自动搜索KAIST数据集 (会显示候选位置让您选择)
- ✅ 自动搜索checkpoint文件 (会推荐最新的)
- ✅ 更新 config_planC_linux.py 中的路径
- ✅ 运行双模态烟雾测试
- ✅ 告诉您训练启动命令

### 步骤3: 启动训练

如果向导成功,直接运行:
```bash
bash train_planC.sh           # 单GPU训练
# 或
bash train_planC.sh 0,1       # 双GPU训练
```

### 步骤4: 监控训练

```bash
# 实时查看loss_macl
tail -f work_dirs/planC_*/train_*.log | grep -E "loss_macl|loss_cls|mAP"

# 检查mAP变化
grep "coco/bbox_mAP" work_dirs/planC_*/train_*.log
```

## 🔧 如果配置向导失败

### 备选方案1: 手动查找路径

```bash
# 运行路径发现脚本
bash find_paths.sh

# 记下输出中的:
# - KAIST数据集实际路径
# - Checkpoint文件实际路径

# 然后运行原始setup脚本
bash setup_planC.sh <数据集路径> <checkpoint路径>
```

### 备选方案2: 诊断环境问题

```bash
# 运行环境检查
bash quick_check.sh

# 查看输出是否有:
# - Python环境问题
# - CUDA不可用
# - MMDetection未安装
```

### 备选方案3: 查看文档

```bash
# 快速修复指南
cat QUICK_FIX.md

# 详细故障排除
cat PATH_TROUBLESHOOTING.md
```

## ⚠️ 常见问题预判

### 问题1: 配置向导找不到数据集

**原因:** KAIST数据集可能不在Linux服务器上

**解决:**
- 选项A: 从Windows传输数据集到Linux
- 选项B: 在Linux上重新下载KAIST数据集
- 选项C: 挂载网络存储 (如果数据集在NAS上)

### 问题2: 配置向导找不到checkpoint

**原因:** 之前的训练是在Windows上进行的

**解决:**
- 选项A: 从Windows传输checkpoint到Linux
  ```bash
  # 在Windows PowerShell:
  scp work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth msi-kklt@server:~/checkpoints/
  ```
- 选项B: 使用COCO预训练模型 (向导会提示此选项)
- 选项C: 从头训练 (不推荐,会花更长时间)

### 问题3: 烟雾测试失败

**可能原因:**
- 数据集结构不正确
- return_modality_pair配置错误
- 缺少annotations文件

**解决:** 查看 PATH_TROUBLESHOOTING.md 的"验证清单"部分

## ✅ 成功标志

配置向导成功后应该看到:

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

训练开始后,在前100行日志中应该看到:
```
loss_macl: 0.3xxx   # 关键!必须>0
loss_cls: 1.xxx
loss_bbox: 0.xxx
```

## 📊 预期训练进展

| Epoch | 预期mAP | 说明 |
|-------|---------|------|
| 1     | 0.53    | Stage 1 baseline水平 |
| 3-5   | 0.55-0.57 | MACL开始起作用 |
| 8-10  | ≥0.60   | **Plan C成功标志** |
| 15    | 0.62-0.64 | 收敛到Plan A水平 |

## 🎯 最终目标

- ✅ loss_macl > 0 (证明双模态加载工作)
- ✅ mAP ≥ 0.60 by epoch 10 (Plan C成功)
- ✅ 训练稳定无崩溃 (Linux环境更可靠)

---

## 📞 需要帮助时

1. **查看日志文件**
   ```bash
   cat work_dirs/planC_*/train_*.log
   ```

2. **收集诊断信息**
   ```bash
   bash quick_check.sh > diagnostic.log
   bash find_paths.sh >> diagnostic.log
   cat diagnostic.log
   ```

3. **检查GPU状态**
   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi  # 实时监控
   ```

---

**现在就将新文件上传到Linux服务器,然后运行 `bash configure_wizard.sh` 开始配置!** 🚀
