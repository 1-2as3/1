#!/bin/bash
# Sync Custom Components to Remote Server
# Usage: bash sync_custom_code.sh user@remote-server:/path/to/mmdetection_remote
# Example: bash sync_custom_code.sh user@192.168.1.100:~/mmdetection_remote

if [ $# -eq 0 ]; then
    echo "Usage: $0 user@remote:/path/to/mmdetection_remote"
    echo "Example: $0 user@192.168.1.100:~/mmdetection_remote"
    exit 1
fi

REMOTE_PATH="$1"
LOCAL_ROOT="."

echo "=========================================="
echo "  Syncing Custom Code to Remote Server"
echo "=========================================="
echo "Remote: $REMOTE_PATH"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "ERROR: rsync not found. Install with: sudo apt-get install rsync"
    exit 1
fi

echo "Step 1: Syncing custom model components..."
echo "-------------------------------------------"

# Sync custom data preprocessors
echo "  - PairedDetDataPreprocessor..."
rsync -avz --progress \
    mmdet/models/data_preprocessors/paired_preprocessor.py \
    "$REMOTE_PATH/mmdet/models/data_preprocessors/" || { echo "ERROR: Failed to sync paired_preprocessor.py"; exit 1; }

# Update data_preprocessors/__init__.py
echo "  - Updating data_preprocessors/__init__.py..."
rsync -avz --progress \
    mmdet/models/data_preprocessors/__init__.py \
    "$REMOTE_PATH/mmdet/models/data_preprocessors/" || { echo "ERROR: Failed to sync __init__.py"; exit 1; }

# Sync custom MACL/DHN/MSP modules
echo "  - MACL/DHN/MSP modules..."
rsync -avz --progress \
    mmdet/models/macldhnmsp/ \
    "$REMOTE_PATH/mmdet/models/macldhnmsp/" || { echo "ERROR: Failed to sync macldhnmsp/"; exit 1; }

# Sync AlignedRoIHead
echo "  - AlignedRoIHead..."
rsync -avz --progress \
    mmdet/models/roi_heads/aligned_roi_head.py \
    "$REMOTE_PATH/mmdet/models/roi_heads/" || { echo "ERROR: Failed to sync aligned_roi_head.py"; exit 1; }

# Sync DomainAligner
echo "  - DomainAligner..."
rsync -avz --progress \
    mmdet/models/utils/domain_aligner.py \
    "$REMOTE_PATH/mmdet/models/utils/" || { echo "ERROR: Failed to sync domain_aligner.py"; exit 1; }

# Sync custom hooks
echo "  - DomainWeightWarmupHook..."
rsync -avz --progress \
    mmdet/engine/hooks/domain_weight_warmup_hook.py \
    "$REMOTE_PATH/mmdet/engine/hooks/" || { echo "ERROR: Failed to sync domain_weight_warmup_hook.py"; exit 1; }

# Update engine/hooks/__init__.py if it exists
if [ -f "mmdet/engine/hooks/__init__.py" ]; then
    rsync -avz --progress \
        mmdet/engine/hooks/__init__.py \
        "$REMOTE_PATH/mmdet/engine/hooks/"
fi

# Sync custom dataset
echo "  - KAISTDataset..."
rsync -avz --progress \
    mmdet/datasets/kaist.py \
    "$REMOTE_PATH/mmdet/datasets/" || { echo "ERROR: Failed to sync kaist.py"; exit 1; }

# Update datasets/__init__.py
rsync -avz --progress \
    mmdet/datasets/__init__.py \
    "$REMOTE_PATH/mmdet/datasets/" || { echo "ERROR: Failed to sync datasets/__init__.py"; exit 1; }

# Sync modified two_stage.py
echo "  - Modified two_stage detector..."
rsync -avz --progress \
    mmdet/models/detectors/two_stage.py \
    "$REMOTE_PATH/mmdet/models/detectors/" || { echo "ERROR: Failed to sync two_stage.py"; exit 1; }

# Sync modified base_sampler.py
echo "  - Modified base_sampler..."
rsync -avz --progress \
    mmdet/models/task_modules/samplers/base_sampler.py \
    "$REMOTE_PATH/mmdet/models/task_modules/samplers/" || { echo "ERROR: Failed to sync base_sampler.py"; exit 1; }

# Update models/__init__.py
echo "  - Updating models/__init__.py..."
rsync -avz --progress \
    mmdet/models/__init__.py \
    "$REMOTE_PATH/mmdet/models/" || { echo "ERROR: Failed to sync models/__init__.py"; exit 1; }

echo ""
echo "Step 2: Syncing configuration files..."
echo "-------------------------------------------"

# Sync all llvip configs
echo "  - Stage1/2/3 configs..."
rsync -avz --progress \
    configs/llvip/ \
    "$REMOTE_PATH/configs/llvip/" || { echo "ERROR: Failed to sync configs/llvip/"; exit 1; }

echo ""
echo "Step 3: Syncing utility scripts..."
echo "-------------------------------------------"

# Sync monitoring scripts
echo "  - Monitoring scripts..."
rsync -avz --progress \
    monitor_training.sh analyze_logs.sh fix_mmcv.sh \
    "$REMOTE_PATH/" 2>/dev/null || echo "  (Optional scripts skipped)"

# Sync documentation
echo "  - Documentation..."
rsync -avz --progress \
    MONITOR_USAGE.md REMOTE_DEPLOY_TROUBLESHOOTING.md \
    "$REMOTE_PATH/" 2>/dev/null || echo "  (Optional docs skipped)"

echo ""
echo "Step 4: Verifying remote installation..."
echo "-------------------------------------------"

# Extract remote host and path
REMOTE_HOST=$(echo "$REMOTE_PATH" | cut -d: -f1)
REMOTE_DIR=$(echo "$REMOTE_PATH" | cut -d: -f2)

echo "  Running remote verification..."
ssh "$REMOTE_HOST" bash << EOF
cd "$REMOTE_DIR"
echo "Python path check:"
python -c "import sys; print('  Python:', sys.executable)"

echo "Checking custom modules..."
python -c "
try:
    from mmdet.models.data_preprocessors import PairedDetDataPreprocessor
    print('  ✓ PairedDetDataPreprocessor')
except Exception as e:
    print('  ✗ PairedDetDataPreprocessor:', e)

try:
    from mmdet.models.roi_heads import AlignedRoIHead
    print('  ✓ AlignedRoIHead')
except Exception as e:
    print('  ✗ AlignedRoIHead:', e)

try:
    from mmdet.models.macldhnmsp import MACLHead
    print('  ✓ MACLHead')
except Exception as e:
    print('  ✗ MACLHead:', e)

try:
    from mmdet.datasets import KAISTDataset
    print('  ✓ KAISTDataset')
except Exception as e:
    print('  ✗ KAISTDataset:', e)

try:
    from mmdet.engine.hooks import DomainWeightWarmupHook
    print('  ✓ DomainWeightWarmupHook')
except Exception as e:
    print('  ✗ DomainWeightWarmupHook:', e)
"

echo ""
echo "Registry check:"
python -c "
from mmdet.registry import MODELS
if 'PairedDetDataPreprocessor' in MODELS.module_dict:
    print('  ✓ PairedDetDataPreprocessor registered in MODELS')
else:
    print('  ✗ PairedDetDataPreprocessor NOT in MODELS registry')
"
EOF

echo ""
echo "=========================================="
echo "  ✓ Sync Complete!"
echo "=========================================="
echo ""
echo "Next steps on remote server:"
echo "  1. cd $REMOTE_DIR"
echo "  2. conda activate mmdet_py311"
echo "  3. python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
echo "          --work-dir work_dirs/stage2_kaist_full_conservative_remote"
