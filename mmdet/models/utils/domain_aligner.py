"""DomainAligner utility module (research skeleton).

This module provides a light-weight alignment helper that can be invoked from
external scripts or integrated later into detectors. It selects a feature map
level (e.g., 'fpn_p3'/'fpn_p4'), optionally normalizes features, and computes
an alignment loss using a pluggable criterion (default: MMDLoss).

Contract:
    - forward(feats: Dict[str, Tensor], modality: Tensor[str/int]) -> Dict[str, Tensor]
    - Expects feats contain keys from FPN like 'p2'/'p3' or 'fpn_p3' depending on model.
    - Returns {'loss_domain': loss}

This is NOT wired into training by default.
"""
from __future__ import annotations

from typing import Dict, Optional, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class DomainAligner(nn.Module):
    def __init__(self,
                 level: str = 'fpn_p3',
                 method: str = 'MMD',
                 loss_weight: float = 0.1,
                 normalize: bool = True,
                 mmd_kernels=(1.0, 2.0, 4.0)):
        super().__init__()
        self.level = level
        self.method = method.upper()
        self.loss_weight = float(loss_weight)
        self.normalize = bool(normalize)
        self.mmd_kernels = tuple(mmd_kernels)

        if self.method == 'MMD':
            self.criterion = MODELS.build(dict(type='MMDLoss', kernels=self.mmd_kernels,
                                               loss_weight=self.loss_weight, normalize=self.normalize))
        else:
            raise NotImplementedError(f'Unsupported domain alignment method: {self.method}')

    def _pick_level(self, feats: Union[Dict[str, Tensor], Sequence[Tensor], Tensor]) -> Tensor:
        """Select one feature level tensor robustly.

        Accepts:
            - Dict[str, Tensor]: standard FPN map
            - Sequence[Tensor]: list/tuple of levels
            - Tensor: already a single level (N,C,H,W)
        """
        if torch.is_tensor(feats):
            return feats
        if isinstance(feats, (list, tuple)):
            # Heuristic: pick middle level
            if len(feats) == 0:
                raise ValueError('Empty feature sequence for DomainAligner')
            return feats[len(feats)//2]
        # Dict branch
        candidates = [self.level, self.level.replace('fpn_', ''), self.level.replace('p', 'fpn_p')]
        for k in candidates:
            if k in feats:
                return feats[k]
        for k in ('p3', 'fpn_p3', 'p4', 'fpn_p4'):
            if k in feats:
                return feats[k]
        return next(iter(feats.values()))

    def _zero_from(self, feats: Union[Dict[str, Tensor], Sequence[Tensor], Tensor]) -> Tensor:
        t: Optional[Tensor] = None
        if torch.is_tensor(feats):
            t = feats
        elif isinstance(feats, (list, tuple)):
            for v in feats:
                if torch.is_tensor(v):
                    t = v
                    break
        elif isinstance(feats, dict):
            for v in feats.values():
                if torch.is_tensor(v):
                    t = v
                    break
        return t.new_tensor(0.0) if t is not None else torch.tensor(0.0)

    def forward(self, feats: Union[Dict[str, Tensor], Sequence[Tensor], Tensor], modality: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute alignment loss between two modality groups.

        Args:
            feats: multi-level features (dict/list/tuple) or a single Tensor (N,C,H,W)
            modality: tensor of shape (N,), values distinguish modalities (e.g., 0/1)
        Returns:
            Dict with {'loss_domain': Tensor}
        """
        try:
            x = self._pick_level(feats)
            n, c, h, w = x.shape
            if modality is None:
                return {'loss_domain': self._zero_from(feats)}
            modality = modality.view(-1)
            if modality.numel() != n:
                return {'loss_domain': self._zero_from(feats)}
            # Global average pool to (N, C)
            pooled = x.mean(dim=(2, 3))
            mask0 = (modality == modality.min())
            mask1 = ~mask0
            if mask0.sum() < 1 or mask1.sum() < 1:
                return {'loss_domain': self._zero_from(feats)}
            loss = self.criterion(pooled[mask0], pooled[mask1])
            return {'loss_domain': loss}
        except Exception:
            # Be fail-safe: return zero so it won't break training
            return {'loss_domain': self._zero_from(feats)}
