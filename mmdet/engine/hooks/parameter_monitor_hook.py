from typing import Optional, Sequence, Union

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from torch.utils.tensorboard import SummaryWriter


class ParameterMonitorHook(Hook):
    """Hook to monitor model parameters during training.
    
    Args:
        log_dir (str): Directory where tensorboard logs will be written.
            Defaults to 'runs/parameter_monitor'.
        interval (int): Logging interval (every k iterations).
            Defaults to 1.
        by_epoch (bool): Whether to log by epoch. Defaults to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 log_dir: str = 'runs/macl_msp_dhn',
                 interval: int = 1,
                 by_epoch: bool = True) -> None:
        self.log_dir = log_dir
        self.interval = interval
        self.by_epoch = by_epoch
        self.writer = None

    def before_train(self, runner: Runner) -> None:
        """Initialize tensorboard writer before training."""
        self.writer = SummaryWriter(self.log_dir)

    def after_train_epoch(self, runner: Runner) -> None:
        """Log parameters after each epoch."""
        if not self.by_epoch or (self.every_n_epochs(runner, self.interval)):
            self._log_parameters(runner)

    def after_train_iter(self, runner: Runner) -> None:
        """Log parameters after each iteration."""
        if self.by_epoch:
            return
        if self.every_n_iters(runner, self.interval):
            self._log_parameters(runner)

    def _log_parameters(self, runner: Runner) -> None:
        """Log model parameters to tensorboard."""
        model = runner.model
        iteration = runner.iter
        epoch = runner.epoch

        # 记录 MSP 模块的 alpha 参数
        if hasattr(model.neck, 'msp_module'):
            alpha = model.neck.msp_module.get_alpha()
            raw_alpha = model.neck.msp_module.alpha.item()
            self.writer.add_scalar('MSP/alpha_sigmoid', alpha, epoch)
            self.writer.add_scalar('MSP/alpha_raw', raw_alpha, epoch)

        # 记录 MACL 的损失
        if hasattr(model.roi_head, 'macl_head'):
            losses = runner.outputs.get('loss', {})
            if 'loss_macl' in losses:
                self.writer.add_scalar('MACL/loss',
                                     losses['loss_macl'].item(), epoch)

        # 记录 DHN sampler 的 queue norm
        if (hasattr(model.roi_head, 'macl_head') and
                hasattr(model.roi_head.macl_head, 'dhn_sampler')):
            queue = model.roi_head.macl_head.dhn_sampler.queue
            queue_norm = queue.norm(dim=1).mean().item()
            self.writer.add_scalar('DHN/queue_norm', queue_norm, epoch)

        # 确保数据被写入
        self.writer.flush()

    def after_run(self, runner: Runner) -> None:
        """Clean up after training."""
        if self.writer is not None:
            self.writer.close()