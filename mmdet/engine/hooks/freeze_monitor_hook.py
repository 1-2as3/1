"""
Training Freeze Monitor Hook

监控训练过程中 Backbone 参数是否真正冻结。
在每个 epoch 开始前检查，确保 requires_grad=False。
"""

from typing import Optional, Sequence
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FreezeMonitorHook(Hook):
    """Monitor frozen parameters during training.
    
    This hook checks if backbone parameters remain frozen throughout training.
    It reports any unexpected trainable parameters in the backbone.
    
    Args:
        check_interval (int): Check every N epochs. Defaults to 1.
        strict (bool): If True, raise error when backbone is not frozen.
            If False, only print warning. Defaults to False.
        print_detail (bool): Whether to print detailed parameter info.
            Defaults to False (only print summary).
    """
    
    priority = 'VERY_LOW'  # Run after other hooks
    
    def __init__(self,
                 check_interval: int = 1,
                 strict: bool = False,
                 print_detail: bool = False):
        self.check_interval = check_interval
        self.strict = strict
        self.print_detail = print_detail
        
        self.initial_check_done = False
    
    def _check_freeze_status(self, runner: Runner) -> dict:
        """Check backbone freeze status.
        
        Returns:
            dict: Status info with keys:
                - backbone_total: Total backbone params
                - backbone_trainable: Trainable backbone params
                - backbone_frozen: Frozen backbone params
                - other_trainable: Trainable non-backbone params
                - is_frozen: Whether backbone is fully frozen
        """
        model = runner.model
        if hasattr(model, 'module'):  # DDP wrapped
            model = model.module
        
        backbone_total = 0
        backbone_trainable = 0
        backbone_frozen = 0
        other_trainable = 0
        trainable_backbone_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_total += param.numel()
                if param.requires_grad:
                    backbone_trainable += param.numel()
                    trainable_backbone_params.append((name, param.numel()))
                else:
                    backbone_frozen += param.numel()
            else:
                if param.requires_grad:
                    other_trainable += param.numel()
        
        is_frozen = (backbone_trainable == 0)
        
        return {
            'backbone_total': backbone_total,
            'backbone_trainable': backbone_trainable,
            'backbone_frozen': backbone_frozen,
            'other_trainable': other_trainable,
            'is_frozen': is_frozen,
            'trainable_params': trainable_backbone_params
        }
    
    def _print_status(self, runner: Runner, status: dict, epoch: int):
        """Print freeze status."""
        runner.logger.info("=" * 60)
        runner.logger.info(f"Freeze Status Check (Epoch {epoch})")
        runner.logger.info("=" * 60)
        
        runner.logger.info(f"Backbone Parameters:")
        runner.logger.info(f"  Total: {status['backbone_total']:,}")
        runner.logger.info(f"  Frozen: {status['backbone_frozen']:,}")
        runner.logger.info(f"  Trainable: {status['backbone_trainable']:,}")
        
        runner.logger.info(f"\nOther Parameters:")
        runner.logger.info(f"  Trainable: {status['other_trainable']:,}")
        
        if status['is_frozen']:
            runner.logger.info(f"\n[OK] Backbone is FULLY FROZEN")
        else:
            runner.logger.warning(
                f"\n[WARNING] Backbone has {status['backbone_trainable']:,} "
                f"trainable parameters!"
            )
            
            if self.print_detail and status['trainable_params']:
                runner.logger.warning("Trainable backbone parameters:")
                for name, count in status['trainable_params'][:10]:
                    runner.logger.warning(f"  - {name}: {count:,} params")
                if len(status['trainable_params']) > 10:
                    remaining = len(status['trainable_params']) - 10
                    runner.logger.warning(f"  ... and {remaining} more")
        
        runner.logger.info("=" * 60)
    
    def before_train(self, runner: Runner):
        """Check freeze status before training starts."""
        runner.logger.info("\n[FreezeMonitorHook] Initial freeze check...")
        status = self._check_freeze_status(runner)
        self._print_status(runner, status, epoch=0)
        
        if not status['is_frozen'] and self.strict:
            raise RuntimeError(
                f"Backbone is not frozen! "
                f"{status['backbone_trainable']:,} trainable parameters found."
            )
        
        self.initial_check_done = True
    
    def before_train_epoch(self, runner: Runner):
        """Check freeze status before each epoch."""
        epoch = runner.epoch
        
        # Skip if not at check interval
        if epoch % self.check_interval != 0:
            return
        
        status = self._check_freeze_status(runner)
        
        # Only print detailed info if status changed or at interval
        if not status['is_frozen']:
            self._print_status(runner, status, epoch)
            
            if self.strict:
                raise RuntimeError(
                    f"Backbone unfrozen at epoch {epoch}! "
                    f"{status['backbone_trainable']:,} trainable parameters."
                )
        else:
            # Backbone still frozen, just log briefly
            runner.logger.info(
                f"[FreezeMonitorHook] Epoch {epoch}: "
                f"Backbone frozen ({status['backbone_frozen']:,} params)"
            )
    
    def after_train(self, runner: Runner):
        """Final freeze check after training."""
        runner.logger.info("\n[FreezeMonitorHook] Final freeze check...")
        status = self._check_freeze_status(runner)
        self._print_status(runner, status, epoch=runner.epoch)
        
        if status['is_frozen']:
            runner.logger.info(
                "[SUCCESS] Backbone remained frozen throughout training!"
            )
        else:
            runner.logger.warning(
                "[WARNING] Backbone was not frozen at the end of training!"
            )
