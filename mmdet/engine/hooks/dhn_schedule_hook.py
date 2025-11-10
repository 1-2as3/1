"""DHN 训练期调度 Hook

按 epoch 动态调整 DHN sampler 的 top_k 和 momentum，实现：
  - 预热阶段 (early epochs)：较小 top_k，快速稳定主表征
  - 中期：提升 top_k，引入更困难负样本
  - 后期：最终 top_k 与较低 momentum，增强多样性

使用方式：在配置中添加 custom_hooks：
  dict(type='DHNScheduleHook', milestones=[3,8,16], topk_stages=[64,96,128,128],
       momentum_stages=[0.995,0.99,0.99,0.99])

若 epoch < milestones[0] 使用 topk_stages[0]；
介于 milestones[i] 与 milestones[i+1] 使用 topk_stages[i+1]；最后阶段使用最后一个。

Compatible with MACLHead + DHNSampler: 通过 roi_head.macl_head.dhn_sampler 访问实例。
"""
from __future__ import annotations

from typing import List
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DHNScheduleHook(Hook):
    priority = 'NORMAL'

    def __init__(self,
                 milestones: List[int] = [3, 8, 16],
                 topk_stages: List[int] = [64, 96, 128, 128],
                 momentum_stages: List[float] = [0.995, 0.99, 0.99, 0.99],
                 verbose: bool = True) -> None:
        assert len(topk_stages) == len(momentum_stages) == len(milestones) + 1, \
            "topk_stages 与 momentum_stages 长度必须比 milestones 多 1"
        self.milestones = milestones
        self.topk_stages = topk_stages
        self.momentum_stages = momentum_stages
        self.verbose = verbose

    def _stage_index(self, epoch: int) -> int:
        for i, m in enumerate(self.milestones):
            if epoch < m:
                return i
        return len(self.milestones)

    def after_train_epoch(self, runner) -> None:
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if not hasattr(model, 'roi_head'):
            return
        roi_head = model.roi_head
        if not hasattr(roi_head, 'macl_head'):
            return
        macl_head = roi_head.macl_head
        if not hasattr(macl_head, 'dhn_sampler') or macl_head.dhn_sampler is None:
            # 可能还未启用 DHN
            return
        sampler = macl_head.dhn_sampler
        idx = self._stage_index(runner.epoch)
        new_topk = self.topk_stages[idx]
        new_momentum = self.momentum_stages[idx]
        # 更新 sampler 参数
        sampler.top_k = min(new_topk, sampler.queue_size)
        sampler.momentum = float(new_momentum)
        if self.verbose:
            runner.logger.info(
                f"[DHNSchedule] epoch={runner.epoch} stage={idx} -> top_k={sampler.top_k}, momentum={sampler.momentum:.4f}" )
