from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LambdaWarmupHook(Hook):
    """Linearly warm up scalar attributes (e.g., roi_head.lambda1) over epochs.

    Config example:
        dict(
            type='LambdaWarmupHook',
            items=[
                dict(path='roi_head.lambda1', start=0.1, target=0.3),
                dict(path='roi_head.lambda2', start=0.05, target=0.1),
            ],
            warmup_epochs=6,
        )
    """

    priority = 'NORMAL'

    def __init__(self, items, warmup_epochs: int = 6, verbose: bool = True):
        assert warmup_epochs > 0
        self.items = items
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def before_train(self, runner):  # type: ignore[override]
        # Initialize current values to "start"
        for it in self.items:
            obj, attr = self._resolve(runner.model, it['path'])
            setattr(obj, attr, float(it['start']))

    def after_train_epoch(self, runner):  # type: ignore[override]
        cur_epoch = runner.epoch + 1
        logger = MMLogger.get_current_instance()
        for it in self.items:
            start = float(it['start']); target = float(it['target'])
            obj, attr = self._resolve(runner.model, it['path'])
            if cur_epoch >= self.warmup_epochs:
                value = target
            else:
                ratio = cur_epoch / self.warmup_epochs
                value = start + (target - start) * ratio
            setattr(obj, attr, value)
            if self.verbose:
                logger.info(f"[LambdaWarmupHook] epoch={cur_epoch} set {it['path']}={value:.4f} (target={target})")

    def _resolve(self, root, path: str):
        parts = path.split('.')
        obj = root
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]
