"""Save a merged, static config for Stage2 (deep merge model keys).
Usage:
  python tools/save_merged_config.py --config configs/llvip/stage2_kaist_domain_ft_nodomain_freezehook.py --out configs/merged/stage2_static.py
"""
import argparse
from pathlib import Path
from mmengine.config import Config
import runpy


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    # Load target config (could be lazy_import)
    cfg = Config.fromfile(args.config)
    # Manually load base model dict by executing python file to avoid lazy mismatch
    try:
        base_ns = runpy.run_path('configs/_base_/models/faster_rcnn_r50_fpn.py')
        base_model = base_ns.get('model', {})
        merged_model = _deep_update(base_model, cfg.get('model', {}))
        cfg.model = merged_model
    except Exception as e:
        print('[WARN] base model merge (exec) failed:', e)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out_path))
    print('[OK] Saved merged config to', out_path)


if __name__ == '__main__':
    main()
