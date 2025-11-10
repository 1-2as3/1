from typing import Optional
from mmengine.hooks import Hook
from mmengine.runner import Runner
import numpy as np


class DomainAdaptationHook(Hook):
    """Hook to dynamically adjust domain adaptation lambda parameter.
    
    The lambda parameter controls the strength of gradient reversal.
    It typically starts at 0 and gradually increases to encourage
    domain-invariant feature learning.
    
    Args:
        initial_lambda (float): Initial lambda value. Defaults to 0.0.
        final_lambda (float): Final lambda value. Defaults to 1.0.
        schedule (str): Schedule type for lambda adjustment.
            - 'linear': Linear increase
            - 'exp': Exponential schedule (lambda = 2/(1+exp(-10*p)) - 1)
            Defaults to 'exp'.
    """
    priority = 'NORMAL'

    def __init__(self,
                 initial_lambda: float = 0.0,
                 final_lambda: float = 1.0,
                 schedule: str = 'exp') -> None:
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.schedule = schedule
        assert schedule in ['linear', 'exp'], \
            f"schedule must be 'linear' or 'exp', got {schedule}"

    def before_train_epoch(self, runner: Runner) -> None:
        """Adjust lambda before each epoch."""
        self._update_lambda(runner)

    def before_train_iter(self, runner: Runner) -> None:
        """Adjust lambda before each iteration (for fine-grained control)."""
        # 如果需要更细粒度的控制，可以在这里更新
        pass

    def _update_lambda(self, runner: Runner) -> None:
        """Update the lambda parameter based on training progress."""
        model = runner.model
        
        # 计算训练进度 (0 到 1)
        max_epochs = runner.max_epochs
        current_epoch = runner.epoch
        progress = current_epoch / max_epochs
        
        # 根据 schedule 计算新的 lambda
        if self.schedule == 'linear':
            lambda_p = self.initial_lambda + \
                      (self.final_lambda - self.initial_lambda) * progress
        elif self.schedule == 'exp':
            # DANN 论文中使用的指数调度
            # lambda_p = 2 / (1 + exp(-10 * progress)) - 1
            lambda_p = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
            # 缩放到 [initial_lambda, final_lambda]
            lambda_p = self.initial_lambda + \
                      (self.final_lambda - self.initial_lambda) * \
                      (lambda_p + 1.0) / 2.0
        
        # 更新模型中的 lambda (如果模型有 domain_classifier)
        if hasattr(model, 'roi_head') and \
           hasattr(model.roi_head, 'domain_classifier'):
            # 将 lambda 存储在 domain_classifier 中供后续使用
            model.roi_head.domain_lambda = lambda_p
            
            # 记录到日志
            runner.logger.info(
                f'Epoch {current_epoch}/{max_epochs}: '
                f'Domain adaptation lambda = {lambda_p:.4f}'
            )

    def after_train_epoch(self, runner: Runner) -> None:
        """Log lambda value after each epoch."""
        model = runner.model
        if hasattr(model, 'roi_head') and \
           hasattr(model.roi_head, 'domain_lambda'):
            lambda_val = model.roi_head.domain_lambda
            runner.message_hub.update_scalar(
                'train/domain_lambda', lambda_val, runner.epoch
            )
