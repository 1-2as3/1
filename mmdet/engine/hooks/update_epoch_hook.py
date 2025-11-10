from mmengine.hooks import Hook
from mmengine.runner import Runner


class UpdateEpochHook(Hook):
    """Hook to update current epoch information in the model.
    
    This hook updates the cur_epoch and max_epochs attributes in the
    model's roi_head, which is used for dynamic loss weight decay.
    """
    priority = 'VERY_LOW'  # Run after other hooks

    def before_train_epoch(self, runner: Runner) -> None:
        """Update epoch information before each training epoch."""
        model = runner.model
        
        # Update epoch information in roi_head
        if hasattr(model, 'roi_head'):
            model.roi_head.cur_epoch = runner.epoch
            model.roi_head.max_epochs = runner.max_epochs
            
            # Log for debugging
            if runner.epoch % 10 == 0:  # Every 10 epochs
                runner.logger.info(
                    f'Updated roi_head epoch info: '
                    f'cur_epoch={runner.epoch}, max_epochs={runner.max_epochs}'
                )
