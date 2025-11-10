"""Hook to hard-freeze backbone parameters by setting requires_grad=False.

This ensures true freezing beyond lr_mult=0. Optionally set BN eval mode.
"""
from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def __init__(self, bn_eval: bool = False, keyword: str = 'backbone') -> None:
        self.bn_eval = bn_eval
        self.keyword = keyword

    def before_train(self, runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Set requires_grad=False for backbone params
        frozen = 0
        for n, p in model.named_parameters():
            if self.keyword in n:
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
        runner.logger.info(f"[FreezeBackboneHook] Hard-froze {frozen} params by requires_grad=False.")

        # Optionally set BN layers to eval mode to stabilize stats
        if self.bn_eval:
            import torch.nn as nn
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
            runner.logger.info("[FreezeBackboneHook] Set BatchNorm layers to eval().")
