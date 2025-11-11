"""Improved preflight check.

Build full Runner (so read_base + custom_imports are honored), fetch one batch,
run forward(loss) under autocast (if fp16 enabled), and report basic stats.
Usage:
  python tools/preflight_check.py --cfg CONFIG_PATH
"""
import argparse
import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    # ensure mmdet registries populated (custom hooks etc.)
    register_all_modules()
    # explicitly import custom modules listed in config custom_imports (if any)
    ci = cfg.get('custom_imports', None)
    if ci:
        import_modules_from_strings(ci.get('imports', []), allow_failed_imports=ci.get('allow_failed_imports', False))
    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.train()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[preflight] Params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M frozen={(total-trainable)/1e6:.2f}M")
    # Rebuild a simple single-worker DataLoader to avoid Windows worker spawn quirks during quick preflight
    from torch.utils.data import DataLoader
    td = runner.train_dataloader
    batch = next(iter(DataLoader(td.dataset, batch_size=td.batch_size, shuffle=False,
                                 collate_fn=td.collate_fn, num_workers=0)))
    device = next(model.parameters()).device
    use_fp16 = bool(getattr(runner, 'fp16', None)) or 'fp16' in cfg
    ctx = torch.cuda.amp.autocast(enabled=use_fp16 and torch.cuda.is_available())
    with ctx:
        out = model.train_step(batch)  # returns dict of losses
    print('[preflight] Loss keys:', list(out.keys()))
    print('[preflight] Batch keys:', list(batch.keys()))
    if 'data_samples' in batch:
        ds0 = batch['data_samples'][0]
        print('[preflight] First sample keys:', ds0._meta_info_keys())
    print('[OK] Preflight passed.')


if __name__ == '__main__':
    main()
