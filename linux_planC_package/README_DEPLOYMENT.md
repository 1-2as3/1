# Plan C Linux Deployment Guide
# ============================================================================

## 📦 Package Contents

```
linux_planC_package/
├── config_planC_linux.py        # Complete training configuration
├── train_planC.sh               # Automated training script
├── test_dual_modality.py        # Smoke test for data loading
├── README_DEPLOYMENT.md         # This file
└── requirements_linux.txt       # Python dependencies
```

## 🚀 Quick Start (3 Steps)

### Step 1: Upload Package to Linux Server

**Using 向日葵 Remote Desktop:**

1. Connect to your Linux machine via 向日葵
2. Transfer the entire `linux_planC_package/` folder to server:
   ```bash
   # On Linux terminal:
   mkdir -p ~/mmdetection_planC
   cd ~/mmdetection_planC
   # Then drag-and-drop the package folder via 向日葵's file transfer
   ```

**Alternative - Using SCP (if SSH available):**
```bash
# On Windows (PowerShell):
scp -r linux_planC_package/ user@server:/home/user/mmdetection_planC/
```

### Step 2: Environment Setup

```bash
# Navigate to mmdetection directory (NOT the package folder!)
cd /path/to/your/mmdetection  # Your existing mmdetection repo

# Copy package files to mmdetection root
cp ~/mmdetection_planC/linux_planC_package/* .

# Verify Python environment
python --version  # Should be 3.8+
python -c "import torch; print(torch.__version__)"  # Should be 1.13+
python -c "import mmcv; print(mmcv.__version__)"    # Should be 2.0+
python -c "import mmdet; print(mmdet.__version__)"  # Should be 3.0+

# Make scripts executable
chmod +x train_planC.sh test_dual_modality.py
```

### Step 3: Configure and Test

**⚠️ CRITICAL: Edit config_planC_linux.py**

Open `config_planC_linux.py` and modify these paths:

```python
# Line 21: Dataset root
data_root = '/path/to/kaist_dataset/'  # Change to your KAIST path

# Line 37: Training annotations
ann_file='annotations/train.txt'  # Verify this path

# Line 53: Validation annotations  
ann_file='annotations/test.txt'   # Verify this path

# Line 243: Pretrained checkpoint
load_from = './work_dirs/stage1_rtmdet_kaist_final/epoch_48.pth'  # Change to your checkpoint
```

**Run Smoke Test:**

```bash
# Test dual-modality data loading (takes ~2 minutes)
python test_dual_modality.py config_planC_linux.py

# Expected output:
#   ✓ Dual-modality OK
#   ✓ All checks passed!
```

If smoke test fails, check:
- Dataset paths in config
- KAIST dataset structure (should have visible/ and lwir/ folders)
- `return_modality_pair=True` in dataset config

## 🎯 Training Commands

### Single GPU Training

```bash
# Simplest command (uses GPU 0 by default)
bash train_planC.sh

# Specify GPU
bash train_planC.sh 1  # Use GPU 1

# Full specification
bash train_planC.sh 0 config_planC_linux.py work_dirs/planC_exp1
```

### Multi-GPU Training (Recommended for faster training)

```bash
# 2 GPUs
bash train_planC.sh 0,1 config_planC_linux.py work_dirs/planC_2gpu

# 4 GPUs
bash train_planC.sh 0,1,2,3 config_planC_linux.py work_dirs/planC_4gpu

# Custom port (if 29500 is occupied)
PORT=29501 bash train_planC.sh 0,1 config_planC_linux.py work_dirs/planC
```

### Monitor Training Progress

```bash
# In a separate terminal (or via 向日葵)
tensorboard --logdir=work_dirs/planC_linux --port=6006

# Then open in browser: http://localhost:6006
# (Use 向日葵's browser forwarding if remote)
```

### Check Training Logs

```bash
# Real-time log monitoring
tail -f work_dirs/planC_linux/train_*.log

# Search for specific metrics
grep "Epoch.*coco/bbox_mAP" work_dirs/planC_linux/train_*.log

# Check for errors
grep -i "error\|exception\|fail" work_dirs/planC_linux/train_*.log
```

## 📊 Expected Training Behavior

### Phase 1: Initialization (First 5 minutes)
```
✓ Loading config...
✓ Building dataset (KAISTDataset)...
✓ Dataset length: 7601 (train) / 2252 (val)
✓ Building model (RTMDet + MACL)...
✓ Loading checkpoint from work_dirs/stage1.../epoch_48.pth
✓ Start training!
```

### Phase 2: Training Loop (Each epoch ~10-15 minutes)
```
Epoch [1][50/1900]  loss: 1.2345  loss_cls: 0.456  loss_bbox: 0.678  loss_macl: 0.112
Epoch [1][100/1900] loss: 1.1234  loss_cls: 0.445  loss_bbox: 0.667  loss_macl: 0.098
...
Epoch [1] complete. Evaluating...
coco/bbox_mAP: 0.5821  coco/bbox_mAP_50: 0.8234  coco/bbox_mAP_75: 0.6123
```

**Key Metrics to Watch:**
- `loss_macl` should start around 0.3-0.5 and decrease to ~0.1
- `coco/bbox_mAP` should improve from ~0.53 (Stage 1) to 0.60+
- If `loss_macl = 0.0000`, dual-modality loading FAILED (re-check config)

### Phase 3: Convergence (After 10-15 epochs)
```
Epoch [12] coco/bbox_mAP: 0.6124  <- Target reached!
[EarlyStopHook] Best mAP: 0.6124 at epoch 12
Training completed successfully.
Best checkpoint: work_dirs/planC_linux/epoch_12.pth
```

## 🔧 Troubleshooting

### Problem 1: "Dataset not found" Error

```bash
# Verify KAIST dataset structure
ls -la /path/to/kaist_dataset/
# Expected:
#   annotations/
#   images/
#     visible/
#     lwir/
```

Fix: Adjust `data_root` and `ann_file` in `config_planC_linux.py`

### Problem 2: loss_macl = 0.0000 (Dual-modality failed)

```bash
# Re-run smoke test with verbose output
python test_dual_modality.py config_planC_linux.py

# Check dataset return_modality_pair setting
grep "return_modality_pair" config_planC_linux.py
# Should show: return_modality_pair=True (TWO occurrences)
```

Fix: Ensure `KAISTDataset._get_paired_data()` is correctly implemented

### Problem 3: CUDA Out of Memory

```bash
# Reduce batch size in config_planC_linux.py
# Line 45: batch_size=4  -> batch_size=2
# Line 54: batch_size=4  -> batch_size=2
```

### Problem 4: Training hangs at initialization

```bash
# Check DataLoader workers
# In config_planC_linux.py:
#   num_workers=6  -> num_workers=2  (reduce if system has limited CPU)

# Or disable multiprocessing temporarily:
#   num_workers=0
```

### Problem 5: Multi-GPU training fails

```bash
# Check NCCL environment
python -c "import torch; print(torch.cuda.nccl.version())"

# Try with explicit master address
MASTER_ADDR=localhost MASTER_PORT=29500 bash train_planC.sh 0,1 config_planC_linux.py work_dirs/planC
```

## 📈 Performance Benchmarks

| Configuration | GPU | Batch Size | Epoch Time | Expected mAP |
|--------------|-----|------------|------------|--------------|
| Single RTX 3090 | 1x | 4 | ~12 min | 0.60-0.62 |
| Dual RTX 3090 | 2x | 8 | ~7 min | 0.60-0.62 |
| Quad V100 | 4x | 16 | ~4 min | 0.62-0.64 |

## 🎓 Advanced Usage

### Modify MACL Parameters

Edit `config_planC_linux.py`, line ~140:

```python
roi_head=dict(
    use_macl=True,
    lambda1=0.01,    # Increase to 0.02 for stronger contrastive learning
    tau=0.07,        # Decrease to 0.05 for harder negatives
    queue_size=1024  # Increase to 2048 for more memory
)
```

### Enable Additional Losses

```python
roi_head=dict(
    use_macl=True,
    lambda1=0.01,      # MACL weight
    lambda2=0.005,     # Enable domain alignment (if multi-domain)
    msp_enabled=True,  # Modality-Specific Pooling
    dhn_enabled=True   # Dynamic Harmonization Network
)
```

### Export Best Checkpoint

```bash
# After training completes
python tools/analysis_tools/analyze_logs.py work_dirs/planC_linux/

# Find best checkpoint
ls -lh work_dirs/planC_linux/epoch_*.pth

# Test best checkpoint
python tools/test.py \
    config_planC_linux.py \
    work_dirs/planC_linux/best_checkpoint.pth \
    --show-dir work_dirs/planC_linux/visualizations
```

## 📞 Support

If you encounter issues:

1. Check smoke test output: `python test_dual_modality.py config_planC_linux.py`
2. Review training logs: `tail -f work_dirs/planC_linux/train_*.log`
3. Verify loss_macl > 0 in first epoch logs
4. Ensure GPT's table requirements are met (fork, num_workers, etc.)

## ✅ Success Criteria

Training is successful if:
- ✅ Smoke test passes (infrared data loaded)
- ✅ loss_macl appears in logs and decreases
- ✅ mAP reaches >= 0.60 within 15 epochs
- ✅ No "Dataset not found" or CUDA OOM errors

Good luck with Plan C on Linux! 🚀
