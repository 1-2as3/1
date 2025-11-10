"""
Mini gradient flow validation using dataset and one batch.
Compatible with MMDet 2.x and 3.x where possible.
"""
import argparse
from pathlib import Path


def log(msg: str):
    print(msg)


def try_imports():
    vers = {}
    try:
        import mmdet
        vers['mmdet'] = getattr(mmdet, '__version__', 'unknown')
    except Exception as e:
        log(f"[ERR] mmdet import failed: {e}")
        return None
    return vers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=str, help='config path')
    p.add_argument('--device', type=str, default='cuda:0')
    args = p.parse_args()

    vers = try_imports()
    if vers is None:
        return

    from mmengine.config import Config
    import copy
    cfg = Config.fromfile(args.config)
    # Ensure full model keys by merging base if missing
    required = ['backbone','rpn_head','train_cfg','test_cfg']
    missing = [k for k in required if k not in cfg.model]
    if missing:
        try:
            from mmengine.config import Config as _C
            base = _C.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
            base_model = copy.deepcopy(base.model)
            # deep merge
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
    # Ensure data_preprocessor exists for v3 style
    if 'data_preprocessor' not in cfg.model:
        cfg.model['data_preprocessor'] = dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32
        )

    # We will rely on synthetic v3 batch for robustness; dataset path often causes failures in research envs
    dataset = None
    dataloader = None
    model = None
    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available() and args.device.startswith('cuda')
        device = args.device if use_cuda else 'cpu'
        # dataset/dataloader
        try:
            from mmdet.datasets import build_dataset, build_dataloader
            ds_cfg = cfg.data['train'] if 'data' in cfg else cfg.train_dataloader.get('dataset')
            dataset = build_dataset(ds_cfg)
            dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
            log('[OK] Built dataset via MMDet 2.x builders')
        except Exception as e:
            log(f"[WARN] build_dataset(v2) failed: {e}")
        # model
        try:
            from mmdet.models import build_detector
            model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            model.init_weights()
            model.to(device)
            model.train()
            log('[OK] Built model via MMDet 2.x build_detector')
        except Exception as e:
            log(f"[WARN] build_detector(v2) failed: {e}")
    except Exception as e:
        log(f"[ERR] torch failure: {e}")
        return

    # Fallback to MMEngine registry (3.x style)
    # Skip dataset building; synthetic batch below ensures deterministic run

    if model is None:
        try:
            from mmdet.registry import MODELS
            model = MODELS.build(cfg.model)
            model.to(device)
            model.train()
            log('[OK] Built model via registry fallback')
        except Exception as e:
            log(f"[ERR] registry model build failed: {e}")
            return

    # Pull one batch and run
    data = None
    if dataloader is not None:
        try:
            data = next(iter(dataloader))
        except Exception as e:
            log(f"[WARN] failed to get batch from real dataloader: {e}")

    # Synthetic fallback batch (paired modalities) if real data missing
    if data is None:
        import torch
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        log('[INFO] Using synthetic fallback batch (random tensor, v3 style).')
        B = 2
        img = torch.randn(B, 3, 640, 640, device=device)  # batched tensor NCHW
        inputs = img  # use batched tensor directly to match v3 forward signature
        data_samples = []
        for i in range(B):
            ds = DetDataSample()
            inst = InstanceData()
            # provide one simple GT box per image to avoid empty-sample fast path
            inst.bboxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=device)
            inst.labels = torch.tensor([0], dtype=torch.long, device=device)
            ds.gt_instances = inst
            # Provide richer metainfo expected by detector components
            ds.set_metainfo({
                'img_shape': (640, 640, 3),
                'ori_shape': (640, 640, 3),
                'pad_shape': (640, 640, 3),
                'batch_input_shape': (640, 640),
                'scale_factor': (1.0, 1.0, 1.0, 1.0),
                'modality': 'visible' if i == 0 else 'infrared'
            })
            data_samples.append(ds)
        data = dict(inputs=inputs, data_samples=data_samples, _style='v3')

    if data is None:
        log('[INFO] No batch available, exit.')
        return

    # Ensure inputs are a single batched tensor (N,C,H,W)
    if isinstance(data.get('inputs', None), list):
        try:
            import torch
            data['inputs'] = torch.stack(data['inputs'], dim=0)
        except Exception:
            pass

    total_loss = None
    try:
        # MMDet 2.x train_step expects a list[dict]
        # Always use mmengine 3.x loss path with synthetic v3 batch
        losses = model(data['inputs'], data.get('data_samples', None), mode='loss')
        if isinstance(losses, dict):
            total = 0.0
            import torch
            for v in losses.values():
                if hasattr(v, 'mean'):
                    total = total + v.mean()
            total_loss = total
        else:
            total_loss = losses if hasattr(losses, 'mean') else None
    except Exception as e:
        log(f"[ERR] forward/backward failed in computing loss: {e}")

    if total_loss is not None:
        try:
            total_loss.backward()
            nz = 0
            allp = 0
            for n,p in model.named_parameters():
                if p.grad is not None:
                    nz += 1
                allp += 1
            log(f"[OK] Backward success. Params with grad: {nz}/{allp}")
        except Exception as e:
            log(f"[ERR] backward failed: {e}")
    else:
        log('[INFO] total_loss is None, no gradients produced.')


if __name__ == '__main__':
    main()
