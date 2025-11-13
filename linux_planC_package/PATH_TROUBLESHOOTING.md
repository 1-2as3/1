# Linux部署路径问题解决方案

## 问题诊断

您遇到的错误:
```
❌ ERROR: Checkpoint not found: ./work_dirs/stage1/epoch_48.pth
❌ ERROR: Data root not found: /data/kaist
```

**原因**: Windows和Linux的文件系统结构不同,Windows开发环境的路径在Linux服务器上不存在。

## 解决步骤

### 第1步: 定位实际文件位置

在Linux服务器上运行路径发现脚本:

```bash
cd ~/xyz/mmdetection/linux_planC_package

# 添加执行权限
chmod +x find_paths.sh quick_check.sh

# 运行环境检查
bash quick_check.sh

# 运行路径发现
bash find_paths.sh
```

### 第2步: 分析输出并找到路径

**查找KAIST数据集:**

脚本会搜索包含以下特征的目录:
- 目录名包含 "kaist" 或 "KAIST"
- 包含 `visible/` 和 `infrared/` 子目录
- 包含 `annotations/` 目录

可能的位置:
- `/data/KAIST/`
- `/mnt/data/kaist_dataset/`
- `/home/msi-kklt/datasets/KAIST/`
- `/home/msi-kklt/data/kaist/`

**查找checkpoint文件:**

脚本会搜索:
- `~/mmdetection/work_dirs/` 下的所有 `.pth` 文件
- 文件名包含 "stage1", "epoch_48", "backup" 的checkpoint

可能的位置:
- `~/mmdetection/work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth`
- `~/xyz/mmdetection/work_dirs/stage1/epoch_48.pth`
- `/home/msi-kklt/checkpoints/stage1_final.pth`

### 第3步: 使用正确路径重新运行setup

找到实际路径后:

```bash
cd ~/xyz/mmdetection/linux_planC_package

# 示例 (请替换为您找到的实际路径)
bash setup_planC.sh /mnt/data/KAIST ~/mmdetection/work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth
```

**setup_planC.sh会自动:**
1. ✅ 验证路径是否存在
2. ✅ 更新 config_planC_linux.py 中的路径
3. ✅ 运行 test_dual_modality.py 烟雾测试
4. ✅ 输出训练启动命令

### 第4步: 验证配置

setup完成后,检查config是否正确更新:

```bash
# 检查data_root
grep "data_root" config_planC_linux.py

# 检查checkpoint路径
grep "load_from" config_planC_linux.py
```

### 第5步: 运行训练

```bash
# 单GPU训练
bash train_planC.sh

# 多GPU训练 (如果有多个GPU)
bash train_planC.sh 0,1
```

## 常见路径场景

### 场景1: 数据集在共享存储

```bash
# 数据集可能在:
/data/shared/datasets/KAIST/
/mnt/nas/datasets/kaist/

# checkpoint可能在:
~/mmdetection/work_dirs/xxx/
```

### 场景2: 数据集在用户home目录

```bash
# 数据集可能在:
/home/msi-kklt/datasets/KAIST/
/home/msi-kklt/data/kaist_dataset/

# checkpoint可能在:
/home/msi-kklt/mmdetection/work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth
```

### 场景3: 需要绝对路径

如果相对路径不工作,使用绝对路径:

```bash
# 获取checkpoint绝对路径
cd ~/xyz/mmdetection/work_dirs/stage1/
pwd  # 显示绝对路径
# 例如: /home/msi-kklt/xyz/mmdetection/work_dirs/stage1/

# 使用绝对路径
bash setup_planC.sh /data/KAIST /home/msi-kklt/xyz/mmdetection/work_dirs/stage1/epoch_48.pth
```

## 如果文件确实不存在

### KAIST数据集不存在

如果Linux服务器上没有KAIST数据集:

**选项1: 从Windows传输**
```bash
# 在Windows PowerShell上:
scp -r C:\path\to\kaist_dataset msi-kklt@server:/data/KAIST
```

**选项2: 在Linux上重新下载**
```bash
# 参考KAIST数据集官方下载方法
# 通常需要先在网站注册并获取下载链接
```

### Checkpoint不存在

如果没有Stage 1的checkpoint:

**选项1: 从Windows传输**
```bash
# 在Windows PowerShell上:
scp C:\Users\Xinyu\mmdetection\work_dirs\stage2_1_pure_detection\stage2_1_backup_ep2.pth msi-kklt@server:~/checkpoints/
```

**选项2: 使用MMDetection预训练模型**

修改 `config_planC_linux.py`:
```python
# 使用COCO预训练的RTMDet
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
```

**选项3: 从头训练 (不推荐)**

注释掉checkpoint加载:
```python
# load_from = None  # 从随机初始化开始
```

## 手动配置方法 (如果setup脚本失败)

如果 setup_planC.sh 无法运行,手动编辑配置:

```bash
# 1. 编辑config
vim config_planC_linux.py

# 2. 修改以下行:
# Line 21: data_root = '/your/actual/kaist/path/'
# Line 243: load_from = '/your/actual/checkpoint.pth'

# 3. 手动运行烟雾测试
python test_dual_modality.py config_planC_linux.py

# 4. 手动启动训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py config_planC_linux.py
```

## 验证清单

在运行训练前,确保:

- [ ] Python环境激活 (mmdet_py311)
- [ ] KAIST数据集路径正确且包含 visible/, infrared/, annotations/
- [ ] Checkpoint文件存在且可读
- [ ] config_planC_linux.py 中的路径已更新
- [ ] test_dual_modality.py 烟雾测试通过
- [ ] GPU可用 (nvidia-smi显示空闲显存)

## 快速诊断命令

```bash
# 检查数据集结构
ls -la /path/to/kaist/  # 应该看到 visible/ infrared/ annotations/

# 检查checkpoint
ls -lh /path/to/checkpoint.pth  # 应该显示文件大小(通常>100MB)

# 检查GPU
nvidia-smi  # 应该显示GPU状态

# 检查Python包
python -c "import mmdet, mmengine, mmcv; print('✓ All packages OK')"

# 测试配置加载
python -c "from mmengine.config import Config; cfg=Config.fromfile('config_planC_linux.py'); print('✓ Config OK')"
```

## 获取帮助

如果问题仍然存在,收集以下信息:

```bash
# 环境信息
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi

# 运行输出
bash quick_check.sh > env_check.log 2>&1
bash find_paths.sh > path_discovery.log 2>&1

# 然后查看日志文件
cat env_check.log
cat path_discovery.log
```
