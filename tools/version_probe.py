"""Version probe: detect mmdet/mmengine/mmcv versions and transforms availability.
Generates logs/version_probe_report.txt and prints concise summary.
"""
from pathlib import Path
import importlib

REPORT = Path('logs/version_probe_report.txt')


def w(msg: str):
    print(msg)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')


def safe_version(mod):
    return getattr(mod, '__version__', 'unknown')


def main():
    REPORT.write_text('', encoding='utf-8')  # clear
    # Probe core packages
    pkgs = ['mmdet', 'mmengine', 'mmcv']
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            w(f'[OK] {p} version={safe_version(m)} path={getattr(m, "__file__", "?")}')
        except Exception as e:
            w(f'[ERR] {p} import failed: {e}')
    # Check transforms registration
    transforms_present = False
    try:
        import mmdet.datasets.transforms as t  # type: ignore
        transforms_present = True
        names = [n for n in dir(t) if 'PackDetInputs' in n]
        w(f'[OK] mmdet.datasets.transforms imported. PackDetInputs symbols={names}')
    except Exception as e:
        w(f'[WARN] mmdet.datasets.transforms missing: {e}')
    # Heuristic recommendation
    if not transforms_present:
        w('[SUGGEST] Environment resembles MMDet 2.x or a partial source clone; use legacy pipeline (DefaultFormatBundle + Collect).')
    else:
        w('[SUGGEST] Full 3.x style pipelines available (PackDetInputs).')

    # Domain align readiness
    # Check if AlignedRoIHead & DomainAligner are registered
    try:
        # Force import to trigger registration
        import mmdet.models.roi_heads.aligned_roi_head  # noqa: F401
        import mmdet.models.utils.domain_aligner  # noqa: F401
        from mmdet.registry import MODELS
        aligned = 'AlignedRoIHead' in MODELS.module_dict
        daligner = 'DomainAligner' in MODELS.module_dict
        w(f'[CHECK] AlignedRoIHead registered={aligned} DomainAligner registered={daligner}')
    except Exception as e:
        w(f'[ERR] registry check failed: {e}')

    w('[DONE] Version probe complete.')

if __name__ == '__main__':
    main()
