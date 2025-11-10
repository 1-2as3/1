"""Apply FreezeBackboneHook and validate with FreezeMonitorHook without full Runner.
Usage:
  python tools/freezehook_runner_probe.py CONFIG
This builds the model, simulates Runner, applies FreezeBackboneHook.before_train,
then runs FreezeMonitorHook.before_train (strict), and reports final stats.
"""
import argparse
import logging
from types import SimpleNamespace

from mmengine.config import Config
import copy


def auto_merge_model(cfg):
    required = ['backbone', 'rpn_head', 'train_cfg', 'test_cfg']
    missing = [k for k in required if k not in cfg.model]
    if missing:
        try:
            from mmengine.config import Config as _C
            base = _C.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
            base_model = copy.deepcopy(base.model)
            def _merge(a,b):
                for k,v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        _merge(a[k], v)
                    else:
                        a[k]=v
            _merge(base_model, cfg.model)
            cfg.model = base_model
            print(f"[INFO] Auto-merged base model fields: {missing}")
        except Exception as e:
            print(f"[WARN] base model auto-merge failed: {e}")


def count_backbone_params(model):
    if hasattr(model, 'module'):
        model = model.module
    total = 0
    trainable = 0
    for n,p in model.named_parameters():
        if 'backbone' in n:
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    return total, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    auto_merge_model(cfg)

    from mmdet.registry import MODELS, HOOKS
    import torch
    model = MODELS.build(cfg.model)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    logger = logging.getLogger('freeze_probe')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    runner = SimpleNamespace(model=model, logger=logger, epoch=0)

    # Build hooks from config (find FreezeBackboneHook and FreezeMonitorHook)
    hooks_cfg = cfg.get('custom_hooks', [])
    freeze_hook = None
    monitor_hook = None
    for hc in hooks_cfg:
        h = HOOKS.build(hc)
        if h.__class__.__name__ == 'FreezeBackboneHook':
            freeze_hook = h
        if h.__class__.__name__ == 'FreezeMonitorHook':
            monitor_hook = h

    if freeze_hook is None:
        freeze_hook = HOOKS.build(dict(type='FreezeBackboneHook', bn_eval=True))
    if monitor_hook is None:
        monitor_hook = HOOKS.build(dict(type='FreezeMonitorHook', check_interval=1, strict=True, print_detail=True))

    # Before
    total, trainable = count_backbone_params(model)
    print(f'[BEFORE] Backbone trainable: {trainable:,}/{total:,}')

    # Apply freeze then monitor (initial check will raise if not frozen and strict)
    freeze_hook.before_train(runner)
    try:
        monitor_hook.before_train(runner)
    except Exception as e:
        print('[MONITOR] strict check raised:', e)

    # After
    total, trainable = count_backbone_params(model)
    print(f'[AFTER] Backbone trainable: {trainable:,}/{total:,}')
    if trainable == 0:
        print('[OK] Backbone frozen by FreezeBackboneHook.')
    else:
        print('[WARN] Backbone still has trainable params.')


if __name__ == '__main__':
    main()
