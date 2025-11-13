#!/bin/bash
# ============================================================================
# Plan C Training Script for Linux
# Usage: bash train_planC.sh [GPU_IDS] [CONFIG] [WORK_DIR]
# Example: bash train_planC.sh 0,1 config_planC_linux.py work_dirs/planC
# ============================================================================

set -e  # Exit on error

# ============== Configuration ==============
GPU_IDS=${1:-"0"}  # Default: single GPU 0
CONFIG=${2:-"config_planC_linux.py"}  # Default config
WORK_DIR=${3:-"work_dirs/planC_linux"}  # Default work directory

# Derived parameters
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${WORK_DIR}/train_${TIMESTAMP}.log"

# ============== Pre-flight Checks ==============
echo "=========================================="
echo "Plan C Training Script"
echo "=========================================="
echo "Configuration: $CONFIG"
echo "GPUs: $GPU_IDS (Total: $NUM_GPUS)"
echo "Work Directory: $WORK_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "ERROR: CUDA is not available"
    exit 1
fi

# Check mmcv installation
if ! python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"; then
    echo "ERROR: mmcv not installed properly"
    exit 1
fi

# Check mmdet installation
if ! python -c "import mmdet; print(f'mmdet: {mmdet.__version__}')"; then
    echo "ERROR: mmdet not installed properly"
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"

echo "All checks passed!"
echo "=========================================="

# ============== Training ==============
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single-GPU training..."
    CUDA_VISIBLE_DEVICES=$GPU_IDS python tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --auto-scale-lr \
        2>&1 | tee "$LOG_FILE"
else
    # Multi-GPU distributed training
    echo "Starting multi-GPU distributed training..."
    PORT=${PORT:-29500}
    CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --launcher pytorch \
        --auto-scale-lr \
        2>&1 | tee "$LOG_FILE"
fi

# ============== Post-training ==============
echo "=========================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "Checkpoints saved to: $WORK_DIR"
echo "=========================================="

# Find best checkpoint by mAP
BEST_CKPT=$(python -c "
import json
import os
from pathlib import Path

work_dir = Path('$WORK_DIR')
scalars_file = work_dir / 'vis_data' / 'scalars.json'

if not scalars_file.exists():
    print('No scalars.json found')
    exit(0)

with open(scalars_file) as f:
    data = json.load(f)

if 'coco/bbox_mAP' not in data:
    print('No mAP data found')
    exit(0)

map_values = data['coco/bbox_mAP']
best_step = max(map_values.items(), key=lambda x: x[1])[0]
best_map = map_values[best_step]

# Find corresponding checkpoint
ckpt_dir = work_dir
ckpts = sorted(ckpt_dir.glob('epoch_*.pth'))
if ckpts:
    best_ckpt = ckpts[-1]  # Last checkpoint (usually best for early stop)
    print(f'{best_ckpt}|{best_map:.4f}')
" 2>/dev/null)

if [ -n "$BEST_CKPT" ]; then
    CKPT_PATH=$(echo "$BEST_CKPT" | cut -d'|' -f1)
    BEST_MAP=$(echo "$BEST_CKPT" | cut -d'|' -f2)
    echo "Best checkpoint: $CKPT_PATH (mAP: $BEST_MAP)"
    
    # Create symlink to best checkpoint
    ln -sf "$(basename $CKPT_PATH)" "$WORK_DIR/best_checkpoint.pth"
    echo "Symlink created: $WORK_DIR/best_checkpoint.pth -> $(basename $CKPT_PATH)"
fi

echo "=========================================="
echo "You can monitor training with:"
echo "  tensorboard --logdir=$WORK_DIR"
echo "=========================================="
