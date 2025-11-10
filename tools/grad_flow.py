import argparse
import torch
from mmengine import Config
from mmdet.registry import MODELS

def grad_flow_synthetic(cfg_path: str, batch: int = 16):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval()
    # 仅对 MACLHead 做合成梯度流测试
    roi_head = getattr(model, 'roi_head', None)
    if roi_head is None or not hasattr(roi_head, 'macl_head'):
        print('[grad_flow] roi_head.macl_head not found')
        return
    head = roi_head.macl_head
    in_dim = getattr(head, 'proj')[0].in_features if hasattr(head, 'proj') else 256
    z_vis = torch.randn(batch, in_dim, requires_grad=True)
    z_ir = torch.randn(batch, in_dim, requires_grad=True)
    out = head(z_vis, z_ir)
    loss = 0.0
    for k,v in out.items():
        if isinstance(v, torch.Tensor):
            loss = loss + v.mean()
    loss.backward()
    print('[grad_flow] Backward ok. Sample gradients:')
    for n, p in head.named_parameters():
        if p.grad is not None:
            print(f'  {n}: grad_mean={p.grad.abs().mean().item():.4e}')
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    grad_flow_synthetic(args.cfg, args.batch)

if __name__ == '__main__':
    main()
