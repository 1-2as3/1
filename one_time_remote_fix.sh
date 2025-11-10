#!/bin/bash
# One-time remote fix script for custom registry/component issues.
# Use AFTER manually copying the local modified repository to remote.
# Steps:
#  1. Remove old site-packages mmdet shadowing your source
#  2. Editable install current source (pip install -e .)
#  3. Validate custom components & registries
#  4. Start training if everything passes
# Usage: bash one_time_remote_fix.sh [WORK_DIR]
# Default WORK_DIR: work_dirs/stage2_kaist_full_conservative_remote

set -e
WORK_DIR="${1:-work_dirs/stage2_kaist_full_conservative_remote}"
CONFIG="configs/llvip/stage2_kaist_full_conservative.py"

ECHO_BLUE='\033[0;34m'
ECHO_GREEN='\033[0;32m'
ECHO_YELLOW='\033[1;33m'
ECHO_RED='\033[0;31m'
ECHO_RESET='\033[0m'

info() { echo -e "${ECHO_BLUE}[INFO]${ECHO_RESET} $*"; }
success() { echo -e "${ECHO_GREEN}[SUCCESS]${ECHO_RESET} $*"; }
warn() { echo -e "${ECHO_YELLOW}[WARN]${ECHO_RESET} $*"; }
error() { echo -e "${ECHO_RED}[ERROR]${ECHO_RESET} $*"; }

info "=== 1. Python / Conda environment check ==="
python -c "import sys; print('Python:', sys.version)" || { error "Python not available"; exit 1; }

if ! python -c "import torch; import mmcv; import mmengine" 2>/dev/null; then
  error "torch/mmcv/mmengine not installed in current environment. Activate env first: conda activate mmdet_py311"; exit 1
fi

info "=== 2. Locate possible shadow mmdet installations ==="
python - <<'PY'
import sys, pathlib
candidates=[]
for p in sys.path:
    mp=pathlib.Path(p)/'mmdet'
    if mp.exists():
        candidates.append(str(mp))
print('Possible mmdet locations:')
for c in candidates:
    print('  ', c)
PY

SITE_PKG_PATH=$(python - <<'PY'
import sys, pathlib
for p in sys.path:
    mp=pathlib.Path(p)/'mmdet'
    if mp.exists() and 'site-packages' in str(mp):
        print(str(mp)); break
PY
)
if [ -n "$SITE_PKG_PATH" ]; then
  warn "Found site-packages mmdet at: $SITE_PKG_PATH (will uninstall)"
  pip uninstall mmdet -y || warn "Uninstall returned non-zero (may already be gone)"
else
  info "No site-packages shadow mmdet found"
fi

info "=== 3. Editable install current source ==="
# Ensure we are at repo root (heuristic: presence of setup.py or README.md)
if [ ! -f setup.py ] && [ ! -f README.md ]; then
  error "Current directory does not look like repo root (no setup.py/README.md). Run: cd /path/to/mmdetection"; exit 1
fi

pip install -e . -v || { error "Editable install failed"; exit 1; }

info "=== 4. Validate custom_imports presence in config ==="
if grep -q "custom_imports" "$CONFIG"; then
  success "custom_imports found in $CONFIG"
else
  warn "custom_imports NOT found in $CONFIG. Will append minimal block." 
  cat >> "$CONFIG" <<'EOF'
# --- Auto-appended custom_imports (one_time_remote_fix.sh) ---
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.models.macldhnmsp',
        'mmdet.models.roi_heads.aligned_roi_head',
        'mmdet.models.utils.domain_aligner',
        'mmdet.engine.hooks.domain_weight_warmup_hook',
        'mmdet.datasets.kaist'
    ],
    allow_failed_imports=False
)
# --------------------------------------------------------------
EOF
  success "Appended custom_imports block"
fi

info "=== 5. Registry & import validation ==="
python - <<'PY'
import importlib
from mmdet.registry import MODELS, DATASETS, HOOKS
failed=[]
# explicit imports
pairs=[
 ('mmdet.models.data_preprocessors.paired_preprocessor','PairedDetDataPreprocessor', MODELS),
 ('mmdet.models.roi_heads.aligned_roi_head','AlignedRoIHead', MODELS),
 ('mmdet.models.macldhnmsp','MACLHead', MODELS),
 ('mmdet.models.utils.domain_aligner','DomainAligner', MODELS),
 ('mmdet.datasets.kaist','KAISTDataset', DATASETS),
 ('mmdet.engine.hooks.domain_weight_warmup_hook','DomainWeightWarmupHook', HOOKS),
]
for mod,name,reg in pairs:
    try:
        m=importlib.import_module(mod)
        obj=getattr(m,name)
        print(f"✓ Imported {name} from {mod}")
    except Exception as e:
        print(f"✗ Import failed {name}: {e}")
        failed.append((name,str(e)))

print('\nRegistry membership:')
for _,name,reg in pairs:
    present = name in reg.module_dict
    print(f" {'✓' if present else '✗'} {name} in {reg.scope}")

if failed:
    print('\n== SUMMARY FAILED ==')
    for n,e in failed:
        print(f"  {n}: {e}")
    import sys; sys.exit(2)
else:
    print('\nAll custom imports succeeded.')
PY
STATUS=$?
if [ $STATUS -ne 0 ]; then
  error "Custom component import/registration FAILED. Abort before training."
  echo "Use: python remote_debug_registry.py for deeper info." 
  exit 2
fi
success "All custom components present."

info "=== 6. Launch training (Stage2 Conservative) ==="
echo "Command: python tools/train.py $CONFIG --work-dir $WORK_DIR"
python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" || { error "Training failed"; exit 3; }

success "Training started successfully. Monitor logs under $WORK_DIR"
exit 0
