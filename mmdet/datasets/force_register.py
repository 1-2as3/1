"""Force import dataset transforms to populate TRANSFORMS registry.

This helps in environments where only partial modules are imported and
pipelines like PackDetInputs are not yet registered at dataset build time.
Safe to import multiple times.
"""
from __future__ import annotations

def _try_imports():
    try:
        # Core 3.x style transforms
        from mmdet.datasets import transforms as _t  # noqa: F401
        # Optionally import individual symbols to ensure side-effects
        from mmdet.datasets.transforms import (  # noqa: F401
            PackDetInputs, LoadImageFromFile, LoadAnnotations, Resize, RandomFlip
        )
    except Exception:
        # Best-effort: some trees may use legacy naming
        try:
            __import__('mmdet.datasets.pipelines')
        except Exception:
            pass

_try_imports()
