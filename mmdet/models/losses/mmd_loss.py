"""Minimal MMD loss implementation (kernel two-sample test) for domain alignment.

This provides a fast, differentiable approximation of Maximum Mean Discrepancy (MMD)
between feature batches from two modalities (e.g., visible vs infrared).

Formula:
    MMD^2(X, Y) = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]

Currently supports Gaussian (RBF) kernel with multiple bandwidths for stability.
Intended for research-stage alignment; not yet integrated into any RoIHead.
Configuration example:
    mmd_cfg = dict(type='MMDLoss', kernels=[1, 2, 4, 8], loss_weight=0.1, normalize=True)

Notes:
    - normalize=True divides by batch size terms to reduce scale drift.
    - Expects input tensors of shape (N, C) after pooling/flatten.
    - If either batch size < 2, returns zero to avoid degenerate kernel stats.
"""
from __future__ import annotations

from typing import Sequence, List
import torch
from torch import Tensor
import torch.nn as nn
import math

from mmdet.registry import MODELS


def _rbf_kernel(x: Tensor, y: Tensor, gamma: float) -> Tensor:
    # Pairwise squared distances
    # (x - y)^2 = x^2 + y^2 - 2xy
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
    dist = x_norm + y_norm - 2 * x @ y.t()
    k = torch.exp(-gamma * dist.clamp(min=0))
    return k


@MODELS.register_module()
class MMDLoss(nn.Module):
    def __init__(self,
                 kernels: Sequence[float] = (1.0, 2.0, 4.0),
                 loss_weight: float = 1.0,
                 normalize: bool = True):
        super().__init__()
        self.kernels: List[float] = list(kernels)
        self.loss_weight = float(loss_weight)
        self.normalize = bool(normalize)

    def forward(self, feat_x: Tensor, feat_y: Tensor) -> Tensor:
        # Flatten to (N, C)
        x = feat_x.view(feat_x.size(0), -1)
        y = feat_y.view(feat_y.size(0), -1)
        nx, ny = x.size(0), y.size(0)
        if nx < 2 or ny < 2:
            return x.new_tensor(0.0)
        mmd_total = 0.0
        for bw in self.kernels:
            # Gamma = 1 / (2 * sigma^2); using bandwidth directly as sigma.
            gamma = 1.0 / (2.0 * (bw ** 2))
            kxx = _rbf_kernel(x, x, gamma)
            kyy = _rbf_kernel(y, y, gamma)
            kxy = _rbf_kernel(x, y, gamma)
            if self.normalize:
                # Remove diagonals for unbiased estimate
                mmd = (kxx.sum() - kxx.diag().sum()) / (nx * (nx - 1))
                mmd += (kyy.sum() - kyy.diag().sum()) / (ny * (ny - 1))
                mmd -= 2.0 * kxy.mean()
            else:
                mmd = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
            mmd_total += mmd
        mmd_total = mmd_total / len(self.kernels)
        return mmd_total * self.loss_weight

    def extra_repr(self) -> str:
        return f"kernels={self.kernels}, loss_weight={self.loss_weight}, normalize={self.normalize}"
