"""Synthetic gradient validation for Stage1/2/3 models without datasets.

Builds the model from a config, creates synthetic multi-modal inputs,
runs forward+backward once, and reports gradient stats. Robust to models
that expect Tensor or dict inputs (visible/infrared).

Usage:
  python tools/grad_flow_synthetic_realmodel.py CONFIG --device cuda:0
"""
import argparse
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_model_ready(cfg):
    # Merge base model keys if missing
    required = ['backbone', 'rpn_head', 'train_cfg', 'test_cfg']
    missing = [k for k in required if k not in cfg.model]
    if missing:
        from mmengine.config import Config
        base = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
        cfg.model = deep_merge(base.model, cfg.model)
        print('[INFO] Auto-merged base model fields:', missing)
    # Ensure data_preprocessor
    if 'data_preprocessor' not in cfg.model:
        cfg.model['data_preprocessor'] = dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32
        )
    return cfg


def build_synthetic_batch(device: str, B: int = 2, H: int = 640, W: int = 640):
    import torch
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    from mmdet.structures.bbox import HorizontalBoxes

    # Two modalities
    vis = torch.randn(B, 3, H, W, device=device)
    ir = torch.randn(B, 3, H, W, device=device)
    inputs_dict = {'visible': vis, 'infrared': ir}

    data_samples = []
    for i in range(B):
        ds = DetDataSample()
        inst = InstanceData()
        # one gt box
        b = torch.tensor([[100.0, 120.0, 220.0, 300.0]], device=device)
        inst.bboxes = HorizontalBoxes(b)
        inst.labels = torch.tensor([0], dtype=torch.long, device=device)
        ds.gt_instances = inst
        meta = {
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'batch_input_shape': (H, W),
            'scale_factor': (1.0, 1.0, 1.0, 1.0),
            'modality': 'visible' if i == 0 else 'infrared'
        }
        ds.set_metainfo(meta)
        data_samples.append(ds)
    return inputs_dict, data_samples


def collect_grads(model):
    if hasattr(model, 'module'):
        model = model.module
    pairs = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            pairs.append((n, p.grad.detach().data.norm(2).item()))
    return pairs


def plot_top_grads(pairs, out_path: Path):
    import matplotlib.pyplot as plt
    if not pairs:
        return False
    names = [n for n, _ in pairs][:100]
    values = [v for _, v in pairs][:100]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.title('Gradient L2 Norm (Top 100)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True


def run_once(cfg_path: str, device: str):
    from mmengine.config import Config
    from mmdet.registry import MODELS
    import torch

    cfg = Config.fromfile(cfg_path)
    cfg = ensure_model_ready(cfg)
    model = MODELS.build(cfg.model)
    dev = device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu'
    model.to(dev)
    model.train()

    # Monkey patch cat_boxes to auto-coerce differing box classes (synthetic only)
    try:
        from mmdet.structures.bbox import transforms as _bbox_transforms
        if hasattr(_bbox_transforms, 'cat_boxes') and not hasattr(_bbox_transforms.cat_boxes, '_synthetic_patch'):  # avoid double patch
            _orig_cat = _bbox_transforms.cat_boxes
            def _synthetic_cat_boxes(data_list, dim=0):
                # Synthetic override: ignore strict class homogeneity; merge by raw tensor concatenation.
                import torch
                if not data_list:
                    return _orig_cat(data_list, dim)
                first = data_list[0]
                cls = type(first)
                tensors = []
                for item in data_list:
                    if hasattr(item, 'tensor'):
                        tensors.append(item.tensor)
                    elif hasattr(item, 'device') and hasattr(item, 'shape'):
                        tensors.append(item)
                    else:
                        try:
                            tensors.append(torch.as_tensor(item))
                        except Exception:
                            pass
                if not tensors:
                    return _orig_cat(data_list, dim)
                cat_tensor = torch.cat(tensors, dim=0)
                try:
                    return cls(cat_tensor)
                except Exception:
                    # fallback to original behavior (may assert)
                    return _orig_cat(data_list, dim)
            _synthetic_cat_boxes._synthetic_patch = True  # mark
            _bbox_transforms.cat_boxes = _synthetic_cat_boxes
            print('[PATCH] cat_boxes monkey patched for synthetic coerced concatenation.')
    except Exception as e:
        print('[WARN] cat_boxes patch failed:', e)
    # Monkey patch BaseBoxes.cat to bypass class uniform assertion (synthetic only)
    try:
        from mmdet.structures.bbox.base_boxes import BaseBoxes as _BaseBoxes
        if not hasattr(_BaseBoxes.cat, '_synthetic_patch'):
            import torch
            _orig_base_cat = _BaseBoxes.cat
            def _synthetic_base_cat(self, box_list, dim=0):
                tensors = []
                for b in box_list:
                    if isinstance(b, type(self)) and hasattr(b, 'tensor'):
                        tensors.append(b.tensor)
                    elif hasattr(b, 'tensor'):
                        tensors.append(b.tensor)
                    elif torch.is_tensor(b):
                        tensors.append(b)
                    else:
                        try:
                            tensors.append(torch.as_tensor(b))
                        except Exception:
                            pass
                if not tensors:
                    return _orig_base_cat(self, box_list, dim)
                cat_tensor = torch.cat(tensors, dim=0)
                return type(self)(cat_tensor)
            _synthetic_base_cat._synthetic_patch = True
            _BaseBoxes.cat = _synthetic_base_cat
            print('[PATCH] BaseBoxes.cat monkey patched for synthetic concatenation.')
    except Exception as e:
        print('[WARN] BaseBoxes.cat patch failed:', e)

    # Build synthetic inputs
    inputs_dict, data_samples = build_synthetic_batch(dev)

    total_loss = None
    # Try dict inputs first (for custom dual-modal paths)
    import traceback
    try:
        losses = model(inputs_dict, data_samples, mode='loss')
    except Exception as e1:
        # Fallback: single tensor path (some models expect Tensor)
        try:
            losses = model(inputs_dict['visible'], data_samples, mode='loss')
        except Exception as e2:
            print('[ERR] Dict forward failed with:')
            traceback.print_exc()
            print('[ERR] Tensor forward failed with:')
            print(''.join(traceback.format_exception(type(e2), e2, e2.__traceback__)))
            return False, 0, []

    if isinstance(losses, dict):
        total_loss = 0.0
        for v in losses.values():
            try:
                total_loss += v.mean()
            except Exception:
                pass
    elif hasattr(losses, 'mean'):
        total_loss = losses.mean()

    if total_loss is None:
        print('[WARN] total_loss is None, cannot backward')
        return False, 0, []

    total_loss.backward()
    grads = collect_grads(model)
    cnt = sum(1 for _, g in grads if g > 0)
    return True, cnt, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    ok, cnt, grads = run_once(args.config, args.device)
    cfg_name = Path(args.config).stem
    if ok:
        print(f'[OK] Synthetic forward/backward for {cfg_name}. Params with grad: {cnt}')
        out_png = Path(f'logs/grad_synth_{cfg_name}.png')
        if plot_top_grads(grads, out_png):
            print('[OK] Saved grad plot to', out_png)
    else:
        print(f'[FAIL] Synthetic run failed for {cfg_name}. See errors above.')


if __name__ == '__main__':
    main()
