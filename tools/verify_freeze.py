"""Verify backbone frozen status for a given config.
Usage:
  python tools/verify_freeze.py CONFIG
"""
import argparse
from mmengine.config import Config
import copy
from mmdet.registry import MODELS
import torch


def count_backbone(model):
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
    # Auto-merge base model keys if missing (handle lazy config)
    required = ['backbone','rpn_head','train_cfg','test_cfg']
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
    model = MODELS.build(cfg.model)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total, trainable = count_backbone(model)
    print(f'Total backbone params: {total:,}')
    print(f'Trainable backbone params: {trainable:,}')
    if trainable == 0:
        print('[OK] Backbone fully frozen (pre-hook). If using FreezeBackboneHook ensure it runs before training.')
    else:
        print('[WARN] Backbone still trainable. Consider adding FreezeBackboneHook or manual requires_grad=False.')

if __name__ == '__main__':
    main()
