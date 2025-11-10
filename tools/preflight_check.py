import argparse
import torch
from mmengine import Config
from mmdet.registry import MODELS

def list_frozen(model):
    total, trainable = 0, 0
    frozen = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen.append(n)
    return total, trainable, frozen

def try_amp_head(model):
    roi_head = getattr(model, 'roi_head', None)
    if roi_head is None or not hasattr(roi_head, 'macl_head'):
        print('[preflight] Skip AMP head test: no macl_head')
        return
    head = roi_head.macl_head
    in_dim = getattr(head.proj[0], 'in_features', 256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    head.to(device)
    z_vis = torch.randn(16, in_dim, device=device)
    z_ir = torch.randn(16, in_dim, device=device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    use_amp = torch.cuda.is_available()
    ctx = torch.cuda.amp.autocast() if use_amp else torch.cpu.amp.autocast(enabled=False)
    with ctx:
        out = head(z_vis, z_ir)
        loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward() if loss.requires_grad else None
    peak = torch.cuda.max_memory_allocated() if device=='cuda' else 0
    print('[preflight] AMP test ok. Loss keys:', list(out.keys()))
    if device=='cuda':
        print('[preflight] CUDA peak memory (bytes):', int(peak))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model)
    model.eval()
    total, trainable, frozen = list_frozen(model)
    print('[preflight] Total params: %.2fM, Trainable: %.2fM' % (total/1e6, trainable/1e6))
    print('[preflight] Frozen tensors:', len(frozen))
    print('[preflight] First 10 frozen names:')
    for n in frozen[:10]:
        print('  -', n)
    try_amp_head(model)

if __name__ == '__main__':
    main()
