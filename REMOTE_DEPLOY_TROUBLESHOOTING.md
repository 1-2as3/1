# Remote Deployment Troubleshooting & Recovery Guide
# Date: 2025-11-10

## Issue 1: numpy 2.x vs pyarrow 15.x conflict
**Error:** `pyarrow 15.0.2 requires numpy<2,>=1.16.6, but you have numpy 2.2.6`

**Root Cause:** PyTorch 2.2 默认拉取最新 numpy（可能是 2.x），但 pyarrow 15.x 要求 <2.0。

**Solution:**
```bash
pip uninstall numpy -y
pip install "numpy<2.0.0,>=1.24.0"
pip install pyarrow==15.0.2
```
或在初始安装时强制顺序：
```bash
pip install "numpy<2.0.0,>=1.24.0" --force-reinstall --no-deps
pip install pyarrow==15.0.2
```

---

## Issue 2: psutil==5.9.8 not found
**Error:** `Could not find a version that satisfies the requirement psutil==5.9.8`

**Root Cause:** 
- 镜像源未同步该版本
- PyPI 已下架 5.9.8（极少见，通常是临时缓存问题）

**Solution:**
1. 使用版本范围代替固定版本：
   ```bash
   pip install "psutil>=5.9.0,<6.0.0"
   ```
2. 或直接用最新稳定版：
   ```bash
   pip install psutil  # 拉取最新（如 6.1.0）
   ```
3. 若坚持 5.9.8，切换到官方 PyPI：
   ```bash
   pip install psutil==5.9.8 --index-url https://pypi.org/simple
   ```

---

## Issue 3: mmcv CUDA ops - No module named 'mmcv._ext'
**Error:** 
- `ModuleNotFoundError: No module named 'mmcv._ext'`
- `undefined symbol: _ZN2at...` 
- `CUDA kernel failed`
- `ImportError: cannot import name 'MultiScaleDeformableAttnFunction'`

**Root Cause:** 
- mmcv 预编译 wheel 与当前 PyTorch/CUDA 版本不匹配
- PyTorch 2.2.0+cu118 与 mmcv 2.0.1 预编译扩展 ABI 不兼容
- 或系统 CUDA toolkit 版本与 wheel 预期不一致

**Diagnosis Steps:**
```bash
# 1. 检查 mmcv 安装路径和版本
python -c "import mmcv; print('mmcv version:', mmcv.__version__); print('mmcv path:', mmcv.__file__)"

# 2. 尝试导入 CUDA 扩展（会显示具体错误）
python -c "from mmcv.ops import nms"

# 3. 检查 _ext 模块是否存在
python -c "import mmcv._ext; print('_ext loaded OK')"

# 4. 检查 PyTorch 与 CUDA 版本
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# 5. 检查是否安装了多个 mmcv 版本冲突
pip list | grep mmcv
```

**Solution 1: 降级到 mmcv 2.0.0（最快，推荐优先尝试）**
```bash
pip uninstall mmcv mmcv-full -y  # 清理所有 mmcv 版本
pip install mmcv==2.0.0

# 验证
python -c "from mmcv.ops import nms; import torch; print('mmcv 2.0.0 OK')"
```

**Solution 2: 使用 mim 自动匹配版本（OpenMMLab 官方工具，强烈推荐）**
```bash
pip install openmim
mim uninstall mmcv  # 清理旧版
mim install mmcv==2.0.1

# mim 会自动检测 PyTorch 版本并下载匹配的 mmcv wheel
# 验证
python -c "from mmcv.ops import nms; print('mim installed mmcv OK')"
```

**Solution 3: 本地编译 mmcv（需要 CUDA toolkit，最兼容但耗时 10-20 分钟）**
```bash
# 安装编译依赖
sudo apt-get update
sudo apt-get install -y build-essential ninja-build

# 安装 CUDA dev toolkit（若系统未安装）
# 方式 A：conda（推荐）
conda install -c conda-forge cudatoolkit-dev=11.8

# 方式 B：系统包管理器（需匹配 CUDA 11.3）
# sudo apt-get install cuda-toolkit-11-3

# 卸载旧版并从源码编译
pip uninstall mmcv mmcv-full -y
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
pip install mmcv==2.0.1 --no-binary mmcv -v

# 验证（编译需要 10-20 分钟，耐心等待）
python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4).cuda(), torch.rand(5).cuda(), 0.5); print('Compiled mmcv OK')"
```

**Solution 4: 降级 PyTorch 到 2.0.1（若 2.2.0 与 mmcv 预编译 wheel 不兼容）**
```bash
pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip uninstall mmcv -y
pip install mmcv==2.0.1

# 验证
python -c "import torch; print('Torch:', torch.__version__)"
python -c "from mmcv.ops import nms; print('mmcv with torch 2.0.1 OK')"
```

**Solution 5: 使用 mmcv 2.1.0（实验性，需 PyTorch >= 2.0）**
```bash
pip uninstall mmcv -y
pip install mmcv==2.1.0

# 注意：mmcv 2.1.x 可能需要 mmdet >= 3.1，检查兼容性
python -c "from mmcv.ops import nms; print('mmcv 2.1.0 OK')"
```

**Recommended Fix Order:**
1. **优先尝试 Solution 2**（mim 自动匹配，成功率最高）
2. 若失败，尝试 Solution 1（降级 mmcv 2.0.0）
3. 若仍失败，尝试 Solution 4（降级 PyTorch 到 2.0.1）
4. 最后才考虑 Solution 3（本地编译，最耗时但最兼容）

**快速修复命令（推荐执行顺序）：**
```bash
# Step 1: 使用 mim（优先）
pip install openmim
mim uninstall mmcv
mim install mmcv==2.0.1
python -c "from mmcv.ops import nms; print('Fixed!')"

# 若 Step 1 失败，执行 Step 2
pip uninstall mmcv -y
pip install mmcv==2.0.0
python -c "from mmcv.ops import nms; print('Fixed with 2.0.0!')"

# 若 Step 2 仍失败，执行 Step 3（降级 PyTorch）
pip uninstall torch torchvision torchaudio mmcv -y
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install mmcv==2.0.1
python -c "from mmcv.ops import nms; print('Fixed with torch 2.0.1!')"
```

---

## Issue 4: Driver version < 525.x
**Error:** `CUDA driver version is insufficient for CUDA runtime version`

**Solution:**
- **优先**：升级驱动到 ≥525.x（支持 CUDA 11.8 runtime）
- **回退**：若无法升级，使用 PyTorch 2.1.x + cu117：
  ```bash
  pip uninstall torch torchvision torchaudio -y
  pip install --index-url https://download.pytorch.org/whl/cu117 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
  ```

---

## Issue 5: OOM (Out of Memory) during training
**Error:** `CUDA out of memory`

**Solution:**
1. 降低 batch_size（在 config 或 `--cfg-options` 中）：
   ```bash
   --cfg-options train_dataloader.batch_size=2
   ```
2. 启用梯度累积（模拟更大 batch）：
   ```python
   # In config: optim_wrapper
   optim_wrapper = dict(
       type='AmpOptimWrapper',  # 若用 AMP
       accumulative_counts=2    # 每2步更新一次
   )
   ```
3. 开启 cudnn_benchmark 与 mixed precision：
   ```python
   env_cfg = dict(cudnn_benchmark=True)
   # 或命令行
   --cfg-options env_cfg.cudnn_benchmark=True
   ```

---

## Issue 6: mAP not improving / stuck at low value
**Symptoms:** Epoch 4–6 mAP < 0.55; losses still decreasing

**Diagnosis:**
- 检查 domain_weight 当前值（可能过高抑制检测）
- 检查数据增强是否过强（Resize scale、ColorJitter 等）
- 检查 RPN proposals 数量是否正常（可通过 log 中 `num_pos_samples` 判断）

**Solution:**
1. 暂停 domain_weight 增长：
   ```bash
   --cfg-options custom_hooks.0.target_domain_weight=0.04  # 冻结在当前值
   ```
2. 降低 MACL 温度（减少表示分散）：
   ```bash
   --cfg-options model.roi_head.macl_head.temperature=0.05
   ```
3. 增加 warmup_iters 或降低初始 LR：
   ```bash
   --cfg-options param_scheduler.warmup_iters=1000 optim_wrapper.optimizer.lr=0.0002
   ```

---

## Issue 7: Training hangs / stuck at DataLoader
**Symptoms:** GPU idle; log 不更新

**Root Cause:** 
- `num_workers` 过高导致死锁（Windows 常见）
- 或 persistent_workers=True 在某些情况下异常

**Solution:**
```bash
--cfg-options train_dataloader.num_workers=0 train_dataloader.persistent_workers=False
```

---

## Issue 8: Checkpoint load failed (KeyError / shape mismatch)
**Error:** `KeyError: 'roi_head.macl_head...'` 或 `size mismatch`

**Root Cause:** Stage1 checkpoint 结构与 Stage2 config 不匹配（如 Stage1 未启用 DHN 但 Stage2 启用）

**Solution:**
1. 确认 Stage1 训练时启用了相同模块（MACL+MSP+DHN）
2. 或在 Stage2 config 中设置 `load_from` 后添加：
   ```python
   load_from = '...'
   resume = False
   # 若有不匹配层，允许部分加载
   custom_hooks.append(dict(type='EMAHook', resume_from=None))
   ```
3. 或手动提取兼容层（高级）：
   ```python
   import torch
   ckpt = torch.load('epoch_21.pth', map_location='cpu')
   # 移除不兼容的 key
   del ckpt['state_dict']['roi_head.macl_head.dhn_queue']
   torch.save(ckpt, 'epoch_21_filtered.pth')
   ```

---

## Quick Recovery Commands

### Reset environment (if corrupted):
```bash
conda deactivate
conda env remove -n mmdet_py311 -y
# Then re-run remote_deploy.sh
```

### Fast dependency re-install (skip PyTorch):
```bash
pip install --no-cache-dir --force-reinstall \
    "numpy<2.0.0,>=1.24.0" mmengine==0.9.1 mmdet==3.3.0

# Fix mmcv using mim
pip install openmim
mim install mmcv==2.0.1
```

### Verify install integrity:
```bash
python -c "import torch, mmcv, mmdet; print('All OK')"
python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4,device='cuda'), torch.rand(5,device='cuda'), 0.5); print('CUDA ops OK')"
```

---

## Contact & Logs
- Save full install log: `bash remote_deploy.sh 2>&1 | tee deploy_$(date +%Y%m%d_%H%M%S).log`
- Save training log: training script 自动生成 `work_dirs/*/YYYYMMDD_HHMMSS.log`
- If issue persists, attach:
  - `pip list > pip_packages.txt`
  - `nvidia-smi > nvidia_info.txt`
  - Last 100 lines of training log
  - Output of: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  - Output of: `python -c "import mmcv; print(mmcv.__version__, mmcv.__file__)"`
