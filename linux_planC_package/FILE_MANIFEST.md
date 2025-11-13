# Linux部署包文件清单 (更新版)

## 📦 完整文件列表 (14个文件)

### 核心配置文件 (必需)
1. **config_planC_linux.py** - 完整训练配置
   - RTMDet + MACL双模态配置
   - 需要更新data_root和load_from路径

2. **train_planC.sh** - 训练启动脚本
   - 自动GPU检测和多GPU支持
   - 内置预检查和日志管理

3. **test_dual_modality.py** - 双模态烟雾测试
   - 验证infrared_img加载
   - 训练前必须通过

### 自动配置工具 (推荐使用)
4. **configure_wizard.sh** ⭐ **最推荐!**
   - 交互式配置向导
   - 自动发现KAIST数据集和checkpoint
   - 自动更新配置文件
   - 自动运行烟雾测试
   - **使用方法:** `bash configure_wizard.sh`

5. **setup_planC.sh** - 原始设置脚本
   - 需要手动提供路径
   - **使用方法:** `bash setup_planC.sh <数据集路径> <checkpoint路径>`

### 诊断工具
6. **find_paths.sh** - 路径发现脚本
   - 搜索KAIST数据集位置
   - 搜索checkpoint文件
   - **使用方法:** `bash find_paths.sh`

7. **quick_check.sh** - 环境快速检查
   - 验证Python/CUDA/MMDetection
   - 显示常见目录结构
   - **使用方法:** `bash quick_check.sh`

### 文档
8. **README_DEPLOYMENT.md** - 主部署文档
   - 完整部署流程
   - 训练命令说明
   - 故障排除指南

9. **QUICK_FIX.md** ⭐ **遇到路径错误先看这个!**
   - 路径问题快速解决方案
   - 3种解决方法
   - 常见路径位置参考

10. **PATH_TROUBLESHOOTING.md** - 详细故障排除
    - 路径问题深度诊断
    - 手动配置方法
    - 各种场景解决方案

11. **QUICK_REFERENCE.txt** - 命令速查表
    - 常用命令快速参考

### 依赖和打包
12. **requirements_linux.txt** - Python依赖列表

13. **create_package.bat** - Windows打包脚本

14. **FILE_MANIFEST.md** - 本文件

---

## 🚀 快速开始 (3步)

### 遇到路径错误时:

```bash
cd ~/xyz/mmdetection/linux_planC_package

# 步骤1: 添加执行权限
chmod +x *.sh *.py

# 步骤2: 运行配置向导 (自动处理一切)
bash configure_wizard.sh

# 步骤3: 启动训练
bash train_planC.sh
```

### 如果向导失败:

```bash
# 手动查找路径
bash find_paths.sh

# 根据输出使用setup
bash setup_planC.sh <找到的数据集路径> <找到的checkpoint路径>

# 启动训练
bash train_planC.sh
```

---

## 📋 文件使用优先级

### 优先级1 (强烈推荐):
- **configure_wizard.sh** - 最简单,自动处理一切

### 优先级2 (需要手动操作):
- **find_paths.sh** + **setup_planC.sh** - 需要手动查找和输入路径

### 优先级3 (诊断用):
- **quick_check.sh** - 环境有问题时运行
- **QUICK_FIX.md** - 遇到错误时参考

### 优先级4 (深度故障排除):
- **PATH_TROUBLESHOOTING.md** - 问题复杂时查阅

---

## 🔄 更新说明

相比原版部署包,新增了:
- ✅ configure_wizard.sh - 交互式自动配置
- ✅ find_paths.sh - 路径自动发现
- ✅ quick_check.sh - 环境诊断
- ✅ QUICK_FIX.md - 快速修复指南
- ✅ PATH_TROUBLESHOOTING.md - 详细故障排除

这些工具解决了"路径在Windows和Linux上不一致"的问题。

---

## ✅ 验证安装

所有文件就位后:

```bash
# 检查文件完整性
ls -1 | wc -l
# 应该显示: 14

# 检查可执行权限
ls -l *.sh
# 应该显示: -rwxr-xr-x (包含x权限)

# 运行快速检查
bash quick_check.sh
# 应该显示Python/CUDA/MMDetection信息
```

---

## 📞 需要帮助?

1. **路径问题** → 查看 `QUICK_FIX.md`
2. **环境问题** → 运行 `quick_check.sh`
3. **训练问题** → 查看 `README_DEPLOYMENT.md`
4. **深度问题** → 查看 `PATH_TROUBLESHOOTING.md`
