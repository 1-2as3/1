#!/bin/bash
# Plan C Linux Auto-Setup Script
# ============================================================================
# This script automates the configuration and validation process
# Usage: bash setup_planC.sh [data_root] [checkpoint_path]
# Example: bash setup_planC.sh /data/kaist ./work_dirs/stage1/epoch_48.pth
# ============================================================================

set -e  # Exit on any error

echo "=================================================="
echo "  Plan C Linux Auto-Setup Script"
echo "=================================================="
echo ""

# ============================================================================
# 1. Parse Arguments
# ============================================================================

DATA_ROOT="${1:-}"
CHECKPOINT="${2:-}"

if [ -z "$DATA_ROOT" ]; then
    echo "⚠️  Data root not provided. Please enter KAIST dataset path:"
    read -p "Data root: " DATA_ROOT
fi

if [ -z "$CHECKPOINT" ]; then
    echo "⚠️  Checkpoint not provided. Please enter Stage 1 checkpoint path:"
    read -p "Checkpoint: " CHECKPOINT
fi

# Expand paths
DATA_ROOT=$(realpath "$DATA_ROOT" 2>/dev/null || echo "$DATA_ROOT")
CHECKPOINT=$(realpath "$CHECKPOINT" 2>/dev/null || echo "$CHECKPOINT")

echo ""
echo "Configuration:"
echo "  Data Root: $DATA_ROOT"
echo "  Checkpoint: $CHECKPOINT"
echo ""

# ============================================================================
# 2. Validate Paths
# ============================================================================

echo "[1/6] Validating paths..."

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ ERROR: Data root not found: $DATA_ROOT"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check KAIST dataset structure
if [ ! -d "$DATA_ROOT/images" ]; then
    echo "❌ ERROR: $DATA_ROOT/images directory not found"
    echo "   Expected structure: $DATA_ROOT/images/visible/ and $DATA_ROOT/images/lwir/"
    exit 1
fi

if [ ! -d "$DATA_ROOT/annotations" ]; then
    echo "❌ ERROR: $DATA_ROOT/annotations directory not found"
    exit 1
fi

echo "✓ Paths validated"

# ============================================================================
# 3. Check Python Environment
# ============================================================================

echo "[2/6] Checking Python environment..."

if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python not found. Please activate your conda/venv environment."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# Check critical packages
echo "  Checking packages..."
python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" || {
    echo "❌ ERROR: PyTorch not installed"
    exit 1
}

python -c "import mmcv; print('  ✓ MMCV:', mmcv.__version__)" || {
    echo "❌ ERROR: MMCV not installed"
    exit 1
}

python -c "import mmdet; print('  ✓ MMDet:', mmdet.__version__)" || {
    echo "❌ ERROR: MMDetection not installed"
    exit 1
}

python -c "import mmengine; print('  ✓ MMEngine:', mmengine.__version__)" || {
    echo "❌ ERROR: MMEngine not installed"
    exit 1
}

# Check CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('  ✓ CUDA:', torch.version.cuda)" || {
    echo "❌ ERROR: CUDA not available"
    exit 1
}

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  ✓ GPUs detected: $NUM_GPUS"

echo "✓ Environment OK"

# ============================================================================
# 4. Update Config File
# ============================================================================

echo "[3/6] Updating config_planC_linux.py..."

if [ ! -f "config_planC_linux.py" ]; then
    echo "❌ ERROR: config_planC_linux.py not found in current directory"
    echo "   Please run this script from the linux_planC_package directory"
    exit 1
fi

# Create backup
cp config_planC_linux.py config_planC_linux.py.backup

# Use Python to update config (more robust than sed)
python - <<EOF
import re

with open('config_planC_linux.py', 'r') as f:
    content = f.read()

# Update data_root
content = re.sub(
    r"data_root = ['\"].*?['\"]",
    f"data_root = '{DATA_ROOT}'",
    content
)

# Update load_from
content = re.sub(
    r"load_from = ['\"].*?['\"]",
    f"load_from = '{CHECKPOINT}'",
    content
)

with open('config_planC_linux.py', 'w') as f:
    f.write(content)

print("✓ Config updated:")
print(f"  data_root = '{DATA_ROOT}'")
print(f"  load_from = '{CHECKPOINT}'")
EOF

echo "✓ Config file updated (backup saved to config_planC_linux.py.backup)"

# ============================================================================
# 5. Run Smoke Test
# ============================================================================

echo "[4/6] Running smoke test..."

if [ ! -f "test_dual_modality.py" ]; then
    echo "⚠️  WARNING: test_dual_modality.py not found, skipping smoke test"
else
    chmod +x test_dual_modality.py
    
    echo ""
    if python test_dual_modality.py config_planC_linux.py; then
        echo ""
        echo "✓ Smoke test passed! Dual-modality data loading confirmed."
    else
        echo ""
        echo "❌ ERROR: Smoke test failed!"
        echo "   Please check:"
        echo "   1. Dataset structure (visible/ and lwir/ folders)"
        echo "   2. Annotation files (train.txt and test.txt)"
        echo "   3. return_modality_pair=True in config"
        exit 1
    fi
fi

# ============================================================================
# 6. Prepare Training Script
# ============================================================================

echo "[5/6] Preparing training script..."

if [ ! -f "train_planC.sh" ]; then
    echo "❌ ERROR: train_planC.sh not found"
    exit 1
fi

chmod +x train_planC.sh
echo "✓ Training script ready"

# ============================================================================
# 7. Create Work Directory
# ============================================================================

echo "[6/6] Creating work directory..."

WORK_DIR="work_dirs/planC_linux_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"
echo "✓ Work directory created: $WORK_DIR"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=================================================="
echo "  ✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Configuration Summary:"
echo "  Dataset: $DATA_ROOT"
echo "  Checkpoint: $CHECKPOINT"
echo "  Work Dir: $WORK_DIR"
echo "  GPUs Available: $NUM_GPUS"
echo ""
echo "Next Steps:"
echo ""
echo "  1️⃣  Single GPU training:"
echo "     bash train_planC.sh 0 config_planC_linux.py $WORK_DIR"
echo ""
echo "  2️⃣  Multi-GPU training (recommended):"
echo "     bash train_planC.sh 0,1 config_planC_linux.py $WORK_DIR"
echo ""
echo "  3️⃣  Monitor progress:"
echo "     tail -f $WORK_DIR/train_*.log"
echo ""
echo "  4️⃣  View in TensorBoard:"
echo "     tensorboard --logdir=$WORK_DIR --port=6006"
echo ""
echo "Expected Training Time:"
echo "  - Single GPU: ~2-3 hours (15 epochs)"
echo "  - Dual GPUs: ~1-1.5 hours"
echo ""
echo "Success Criteria:"
echo "  ✓ loss_macl appears in logs and decreases"
echo "  ✓ mAP improves from ~0.53 to 0.60+"
echo "  ✓ Training completes without OOM errors"
echo ""
echo "=================================================="
echo "  🚀 Ready to train! Good luck!"
echo "=================================================="
