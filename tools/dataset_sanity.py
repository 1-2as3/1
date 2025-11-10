"""Dataset sanity checks for KAIST (and optional LLVIP/M3FD).
Reports: sample counts, empty annotations, duplicate image IDs, modality distribution.
"""
import argparse
from pathlib import Path
from collections import Counter

from mmengine.config import Config
from importlib import import_module

# Force import transforms to ensure registry registration in this env
try:
    from mmdet.datasets.transforms import (
        PackDetInputs, LoadImageFromFile, LoadAnnotations, Resize, RandomFlip  # noqa: F401
    )
except Exception as _e:
    print('[WARN] Transforms force-import failed:', _e)

# Broader attempts for different source layouts (e.g., pipelines alias)
for _mod in ('mmdet.datasets.transforms', 'mmdet.datasets.pipelines'):
    try:
        import_module(_mod)
        print(f'[INFO] Imported {_mod} for transform registration')
    except Exception as _e:
        print(f'[WARN] Could not import {_mod}:', _e)

# Ensure default scope and full module registration
try:
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    print('[INFO] register_all_modules(init_default_scope=True) done')
except Exception as _e:
    print('[WARN] register_all_modules failed:', _e)

# Verify and patch-register PackDetInputs if needed
try:
    from mmengine.registry import TRANSFORMS
    from mmdet.datasets.transforms import PackDetInputs as _PackDetInputs
    if 'PackDetInputs' not in TRANSFORMS.module_dict:
        TRANSFORMS.register_module()(_PackDetInputs)
        print('[INFO] Patched registration for PackDetInputs into TRANSFORMS')
    else:
        print('[INFO] PackDetInputs already in TRANSFORMS registry')
except Exception as _e:
    print('[WARN] Could not verify/register PackDetInputs:', _e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Config file path')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of samples to scan (0=all)')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    # Try to locate train dataset config
    if 'data' in cfg and 'train' in cfg.data:
        ds_cfg = cfg.data['train']
    else:
        ds_cfg = cfg.get('train_dataloader', {}).get('dataset', None)
    if ds_cfg is None:
        print('[ERR] No train dataset config found.')
        return

    # Build dataset via registry fallback
    from mmdet.registry import DATASETS
    try:
        dataset = DATASETS.build(ds_cfg)
        try:
            print('[INFO] Dataset built. length =', len(dataset))
        except Exception:
            print('[INFO] Dataset built. length = (unknown)')
        print('[INFO] Begin scanning samples (this may read images).')
    except Exception as e:
        print('[ERR] dataset build failed:', e)
        # Fallback: lightweight ann_file scan
        ann_file = ds_cfg.get('ann_file') if isinstance(ds_cfg, dict) else None
        if ann_file and Path(ann_file).exists():
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                limit = args.limit or len(lines)
                total = min(len(lines), limit)
                print('=== Dataset Sanity (fallback) ===')
                print('Ann file:', ann_file)
                print('Total IDs in ann_file:', len(lines))
                print('Scanned (limited):', total)
                print('[NOTE] Pipeline未注册，执行轻量级统计；详细空标注/模态分布需pipeline正常后再运行。')
            except Exception as e2:
                print('[ERR] fallback ann_file scan failed:', e2)
        return

    total = 0
    empty_ann = 0
    modalities = Counter()
    ids = []

    for idx, item in enumerate(dataset):
        if args.limit and idx >= args.limit:
            break
        total += 1
        data_samples = item.get('data_samples') if isinstance(item, dict) else None
        if data_samples is None:
            continue
        # MMDet3 style: data_samples may be list or single
        if isinstance(data_samples, list):
            for ds in data_samples:
                _proc(ds, modalities, ids, lambda: empty_ann.__iadd__(1))
        else:
            _proc(data_samples, modalities, ids, lambda: empty_ann.__iadd__(1))

    dup = len(ids) - len(set(ids))
    print('=== Dataset Sanity Report ===')
    print('Total scanned samples:', total)
    print('Empty annotation samples:', empty_ann)
    print('Duplicate img_ids:', dup)
    print('Modality distribution:', dict(modalities))
    if total > 0:
        print('Empty ann ratio: {:.2f}%'.format(100.0 * empty_ann / total))
    if dup > 0:
        print('[WARN] Found duplicate image IDs, please inspect dataset indexing.')


def _proc(data_sample, modalities: Counter, ids_list, empty_incr):
    # meta
    mid = getattr(data_sample, 'metainfo', {}).get('img_id', None)
    if mid is not None:
        ids_list.append(mid)
    modality = getattr(data_sample, 'metainfo', {}).get('modality', 'unknown')
    modalities[modality] += 1
    # instances
    gt_instances = getattr(data_sample, 'gt_instances', None)
    if gt_instances is None or len(getattr(gt_instances, 'bboxes', [])) == 0:
        empty_incr()


if __name__ == '__main__':
    main()
