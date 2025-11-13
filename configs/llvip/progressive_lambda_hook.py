"""
Progressive MACL Lambda Hook
=============================
动态调整 lambda1 权重,实现渐进式MACL warmup

Schedule:
- Epoch 1-2: lambda1 = 0.005  (轻量级,稳定特征空间)
- Epoch 3-4: lambda1 = 0.0075 (1.5x增强语义对齐)
- Epoch 5-6: lambda1 = 0.01125 (1.5x进一步增强)
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ProgressiveLambdaHook(Hook):
    """
    Hook to progressively increase MACL lambda1 during training.
    
    Args:
        milestones (list): Epochs at which to increase lambda1
        gamma (float): Multiplication factor for lambda1
        initial_lambda (float): Starting lambda1 value
    """
    
    def __init__(self, 
                 milestones=[3, 5],  # Epoch 3和5时增加
                 gamma=1.5,
                 initial_lambda=0.005):
        self.milestones = milestones
        self.gamma = gamma
        self.initial_lambda = initial_lambda
        self.current_lambda = initial_lambda
        
    def before_train_epoch(self, runner):
        """在每个epoch开始前调整lambda1"""
        epoch = runner.epoch
        
        # 计算当前应该的lambda值
        num_increases = sum([1 for m in self.milestones if epoch >= m])
        target_lambda = self.initial_lambda * (self.gamma ** num_increases)
        
        # 如果lambda需要更新
        if abs(target_lambda - self.current_lambda) > 1e-6:
            old_lambda = self.current_lambda
            self.current_lambda = target_lambda
            
            # 更新模型的lambda1参数
            if hasattr(runner.model, 'module'):
                # DDP模式
                model = runner.model.module
            else:
                model = runner.model
                
            if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'lambda1'):
                model.roi_head.lambda1 = self.current_lambda
                runner.logger.info(
                    f'\n{"="*80}\n'
                    f'Progressive Lambda Update at Epoch {epoch}\n'
                    f'Old lambda1: {old_lambda:.6f} → New lambda1: {self.current_lambda:.6f}\n'
                    f'MACL weight increased by {self.gamma}x\n'
                    f'{"="*80}\n'
                )
            else:
                runner.logger.warning(
                    f'Cannot find lambda1 attribute in model.roi_head. '
                    f'Make sure your RoIHead supports MACL.'
                )
    
    def after_train_epoch(self, runner):
        """在每个epoch结束后记录lambda值"""
        runner.logger.info(
            f'Epoch {runner.epoch} completed with lambda1={self.current_lambda:.6f}'
        )
