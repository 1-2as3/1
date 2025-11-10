"""Remote registry & path diagnostic script.
Run: python remote_debug_registry.py
Outputs:
  - Python path entries
  - Actual mmdet package path & version
  - Presence of custom classes in registries
  - Potential duplicate installs
"""
import sys
import importlib
from pathlib import Path

print("=== Python sys.path (ordered) ===")
for p in sys.path:
    print(" ", p)
print()

print("=== Checking mmdet import origin ===")
try:
    import mmdet
    import inspect
    print("mmdet module file:", inspect.getfile(mmdet))
    print("mmdet package path:", Path(mmdet.__file__).parent)
except Exception as e:
    print("ERROR: cannot import mmdet:", e)
    sys.exit(1)
print()

print("=== Checking mmengine/mmcv versions ===")
try:
    import mmengine, mmcv
    print("mmengine:", mmengine.__version__)
    print("mmcv:", mmcv.__version__)
except Exception as e:
    print("ERROR importing mmengine/mmcv:", e)
print()

# Attempt to import custom modules explicitly
modules = [
    ("mmdet.models.data_preprocessors.paired_preprocessor", "PairedDetDataPreprocessor"),
    ("mmdet.models.roi_heads.aligned_roi_head", "AlignedRoIHead"),
    ("mmdet.models.macldhnmsp", "MACLHead"),
    ("mmdet.models.utils.domain_aligner", "DomainAligner"),
    ("mmdet.engine.hooks.domain_weight_warmup_hook", "DomainWeightWarmupHook"),
    ("mmdet.datasets.kaist", "KAISTDataset"),
]

print("=== Explicit import of custom modules ===")
custom_errors = []
for mod, symbol in modules:
    try:
        m = importlib.import_module(mod)
        obj = getattr(m, symbol)
        print(f" ✓ {symbol} from {mod}")
    except Exception as e:
        print(f" ✗ {symbol} FAILED: {e}")
        custom_errors.append((symbol, str(e)))
print()

print("=== Registry membership check ===")
try:
    from mmdet.registry import MODELS, DATASETS, HOOKS
    def check(name, reg):
        return name in reg.module_dict
    checks = [
        ("PairedDetDataPreprocessor", MODELS),
        ("AlignedRoIHead", MODELS),
        ("MACLHead", MODELS),
        ("DomainAligner", MODELS),
        ("KAISTDataset", DATASETS),
        ("DomainWeightWarmupHook", HOOKS),
    ]
    for name, reg in checks:
        print(f" {'✓' if check(name, reg) else '✗'} {name} in {reg.scope} registry")
except Exception as e:
    print("ERROR checking registries:", e)
print()

print("=== Searching for duplicate mmdet installations ===")
# Look for any path entries containing '/site-packages/mmdet'
candidate_paths = []
for p in sys.path:
    try:
        pkg_path = Path(p) / 'mmdet'
        if pkg_path.exists():
            candidate_paths.append(str(pkg_path))
    except Exception:
        pass
for cp in candidate_paths:
    print(" ->", cp)
print()

if custom_errors:
    print("=== SUMMARY: Some custom imports failed ===")
    for symbol, err in custom_errors:
        print(f"  {symbol}: {err}")
    print("  Possible causes: not synced, wrong working directory, old site-packages version overshadowing local edits.")
    print("Suggested next steps:\n  1) pip uninstall mmdet\n  2) cd to repository root\n  3) pip install -e .\n  4) Re-run this script")
else:
    print("All custom imports succeeded.")

print("Done.")
