#!/bin/bash
# Remote Deployment Script: MMDetection Stage2 Training Environment
# Target: RTX 4090 + CUDA 11.3 + cuDNN 8.6 + PyTorch 2.2.0 cu118
# Date: 2025-11-10

set -e  # Exit on error

echo "=== Step 1: Driver & CUDA Check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed. Check driver installation."; exit 1; }
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
echo "Driver version: $DRIVER_VERSION (require >= 525.x for cu118)"

echo ""
echo "=== Step 2: Create Conda Environment (Python 3.11) ==="
conda create -n mmdet_py311 python=3.11 -y || { echo "ERROR: conda create failed"; exit 1; }
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmdet_py311

echo ""
echo "=== Step 3: Configure pip (Optional: use Tsinghua mirror) ==="
# Uncomment below if in China mainland; else skip
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=== Step 4: Install PyTorch 2.2.0+cu118 ==="
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 || { echo "ERROR: PyTorch install failed"; exit 1; }

echo ""
echo "=== Step 5: Verify PyTorch CUDA ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available!"
print('Device count:', torch.cuda.device_count(), 'Name:', torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Step 6: Install Core Dependencies (NumPy<2 constraint) ==="
pip install --no-cache-dir "numpy<2.0.0,>=1.24.0" scipy==1.11.4 pillow==10.3.0 \
    opencv-python==4.8.1.78 matplotlib==3.8.4 seaborn==0.13.2 \
    scikit-image==0.22.0 scikit-learn==1.4.2 tqdm==4.66.4 pandas==2.2.2 \
    pyarrow==15.0.2 || { echo "ERROR: Core scientific packages failed"; exit 1; }

echo ""
echo "=== Step 7: Install OpenMMLab Stack ==="
echo "Installing mmengine first..."
pip install --no-cache-dir mmengine==0.9.1 || { echo "ERROR: mmengine install failed"; exit 1; }

echo "Installing mmcv using mim (automatic version matching)..."
pip install --no-cache-dir openmim || { echo "ERROR: openmim install failed"; exit 1; }
mim install mmcv==2.0.1 || {
    echo "WARNING: mim install mmcv 2.0.1 failed; trying direct pip install mmcv 2.0.0"
    pip install --no-cache-dir mmcv==2.0.0 || {
        echo "ERROR: All mmcv install methods failed. See REMOTE_DEPLOY_TROUBLESHOOTING.md Issue 3"
        exit 1
    }
}

echo "Installing mmdet and mmpretrain..."
pip install --no-cache-dir mmdet==3.3.0 mmpretrain==1.2.0 || { echo "ERROR: mmdet install failed"; exit 1; }

echo ""
echo "=== Step 8: Install Detection & Geometry Utils ==="
pip install --no-cache-dir shapely==2.0.4 pycocotools==2.0.7 lap==0.4.0 addict==2.4.0 \
    pyyaml==6.0.1 packaging==24.0 yapf==0.40.2 termcolor==2.4.0 Jinja2==3.1.4

echo ""
echo "=== Step 9: Install Augmentation & Vision Extras ==="
pip install --no-cache-dir albumentations==1.4.4 imgaug==0.4.0 timm==0.9.16 \
    einops==0.7.0 fvcore==0.1.5.post20221221 omegaconf==2.3.0 filterpy==1.4.5

echo ""
echo "=== Step 10: Install Logging, Stats & Misc ==="
pip install --no-cache-dir tensorboard==2.16.2 rich==13.7.1 pot==0.9.3 \
    "psutil>=5.9.0,<6.0.0" future==1.0.0 requests==2.32.3

echo ""
echo "=== Step 11: Verify mmcv CUDA ops ==="
python - <<'PY'
import mmengine, mmcv, mmdet
from mmcv.ops import nms
import torch
x = torch.randn(10, 4, device='cuda')
scores = torch.rand(10, device='cuda')
keep = nms(x, scores, 0.5)
print('mmcv.ops.nms OK, kept bboxes:', keep.shape[0])
PY

echo ""
echo "=== Step 12: Clone/Update MMDetection Repo ==="
# Adjust repo URL to your actual git remote
REPO_URL="https://github.com/<YOUR_ORG>/mmdetection.git"  # Replace with your actual URL
TARGET_DIR="mmdetection_remote"

if [ -d "$TARGET_DIR" ]; then
    echo "Repository exists, pulling latest..."
    cd $TARGET_DIR
    git pull origin main || git pull origin master
else
    echo "Cloning repository..."
    git clone $REPO_URL $TARGET_DIR
    cd $TARGET_DIR
fi

echo ""
echo "=== Step 13: Upload Stage1 Checkpoint (if needed) ==="
# Example: scp from local to remote
# scp C:/Users/Xinyu/mmdetection/work_dirs/stage1_longrun_full/epoch_21.pth user@remote:~/mmdetection_remote/work_dirs/stage1_longrun_full/
echo "Ensure work_dirs/stage1_longrun_full/epoch_21.pth exists on remote before training."
echo "If not present, upload via scp or rsync."

echo ""
echo "=== Step 14: Quick Sanity Test (Inference) ==="
# Uncomment below if epoch_21.pth is present and you want a quick test
# python tools/test.py configs/llvip/stage2_llvip_kaist_finetune.py \
#     work_dirs/stage1_longrun_full/epoch_21.pth \
#     --show-dir quick_vis --cfg-options model.roi_head.use_domain_aligner=False

echo ""
echo "=== Environment Setup Complete! ==="
echo "To activate: conda activate mmdet_py311"
echo "To train Stage2 (conservative plan A):"
echo "  python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
echo "         --work-dir work_dirs/stage2_kaist_full_conservative_remote"
echo ""
echo "Monitor logs:"
echo "  grep -E 'pascal_voc/mAP|overall_loss|domain_weight' \\"
echo "       work_dirs/stage2_kaist_full_conservative_remote/*.log | tail -n 50"
echo ""
echo "Branch to Plan C (if epoch6 mAP < 0.60):"
echo "  python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
echo "         --work-dir work_dirs/stage2_kaist_full_C \\"
echo "         --cfg-options custom_hooks.0.target_domain_weight=0.06 \\"
echo "                       custom_hooks.0.warmup_epochs=6 \\"
echo "                       model.roi_head.macl_head.temperature=0.06"
echo ""
echo "Branch to Plan B (if epoch12 mAP >= 0.70):"
echo "  python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
echo "         --work-dir work_dirs/stage2_kaist_full_B \\"
echo "         --cfg-options optim_wrapper.optimizer.lr=0.00035 \\"
echo "                       custom_hooks.0.target_domain_weight=0.12 \\"
echo "                       model.backbone.frozen_stages=1"

