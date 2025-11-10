"""
轻量级 MACL 对齐质量指标 Hook（不依赖 sklearn/scipy）
在每个 epoch 结束时，直接使用 MACLHead.latest_embeddings 计算：
- 可见/红外类内平均欧氏距离
- 跨模态类间平均欧氏距离
- 对齐分数：inter / (intra_vis + intra_ir + eps)
"""
from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch


@HOOKS.register_module()
class MACLAlignmentHook(Hook):
    """基于 embedding 空间的对齐指标记录 Hook。

    Args:
        interval (int): 记录间隔（按 epoch）。默认 1
        eps (float): 数值稳定常数。默认 1e-6
    """

    priority = 'NORMAL'

    def __init__(self, interval: int = 1, eps: float = 1e-6) -> None:
        self.interval = int(interval)
        self.eps = float(eps)

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        if epoch % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None or not hasattr(roi_head, 'macl_head'):
            return

        macl_head = roi_head.macl_head
        embeds = getattr(macl_head, 'latest_embeddings', None)
        if not isinstance(embeds, dict):
            return

        vis = embeds.get('vis', None)
        ir = embeds.get('ir', None)
        if vis is None or ir is None:
            return

        # 统一到 cpu 计算，避免显存开销
        if torch.is_tensor(vis):
            vis = vis.detach().float().cpu()
        if torch.is_tensor(ir):
            ir = ir.detach().float().cpu()

        n_vis = vis.shape[0]
        n_ir = ir.shape[0]
        if n_vis == 0 or n_ir == 0:
            return

        # pairwise 平均欧氏距离
        def mean_pairwise_euclidean(x: torch.Tensor) -> float:
            if x.shape[0] < 2:
                return 0.0
            # torch.cdist 计算 O(N^2) 距离矩阵
            d = torch.cdist(x, x, p=2)
            return float(d.mean().item())

        vis_intra = mean_pairwise_euclidean(vis)
        ir_intra = mean_pairwise_euclidean(ir)
        inter = float(torch.cdist(vis, ir, p=2).mean().item())

        align_score = inter / (vis_intra + ir_intra + self.eps)

        runner.logger.info(
            f"[MACLAlignmentHook] Inter: {inter:.4f}, "
            f"Intra(vis/ir): {vis_intra:.4f}/{ir_intra:.4f}, "
            f"AlignScore: {align_score:.4f}"
        )
