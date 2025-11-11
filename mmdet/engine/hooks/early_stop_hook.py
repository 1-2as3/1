from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EarlyStopHook(Hook):
    """Early stop training when validation mAP stays below a threshold.

    Args:
        metric (str): Metric key recorded by evaluator. Default 'pascal_voc/mAP'.
        threshold (float): Stop if metric < threshold for ``patience`` epochs.
        patience (int): Number of consecutive epochs allowed under threshold.
        begin (int): Enable hook from this epoch (1-based). Default 1.
        verbose (bool): Whether to log decisions.
    """

    priority = 'VERY_HIGH'

    def __init__(
        self,
        metric: str = 'pascal_voc/mAP',
        threshold: float = 0.55,
        patience: int = 3,
        begin: int = 1,
        verbose: bool = True,
    ) -> None:
        self.metric = metric
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.begin = int(begin)
        self.verbose = verbose
        self._bad_epochs = 0

    def after_val_epoch(self, runner, metrics=None) -> None:  # type: ignore[override]
        """Callback after validation epoch.

        Args:
            runner (Runner): The runner instance.
            metrics (dict | None): Metrics dict passed by evaluator (if available).
        """
        # Skip before begin epoch
        cur_epoch = runner.epoch + 1  # 1-based display
        if cur_epoch < self.begin:
            return

        logger = MMLogger.get_current_instance()

        # 1) Try from passed-in metrics first
        value = None
        if isinstance(metrics, dict):
            if self.metric in metrics:
                value = metrics[self.metric]
            elif f'val/{self.metric}' in metrics:
                value = metrics[f'val/{self.metric}']
            elif self.metric.endswith('/mAP') and 'mAP' in metrics:
                value = metrics['mAP']

        # 2) Fallback: query message hub scalars
        if value is None:
            val_names = [f'val/{self.metric}', self.metric, 'mAP', 'val/mAP']
            for name in val_names:
                try:
                    scalar = runner.message_hub.get_scalar(name)
                    if scalar is not None:
                        value = scalar.current()
                except Exception:
                    continue
                if value is not None:
                    break

        if value is None:
            if self.verbose:
                avail_keys = list(metrics.keys()) if isinstance(metrics, dict) else []
                logger.warning(
                    f"[EarlyStopHook] Metric '{self.metric}' not found this epoch, skip check. "
                    f"Available metrics keys: {avail_keys}")
            return

        if value < self.threshold:
            self._bad_epochs += 1
            if self.verbose:
                logger.info(
                    f"[EarlyStopHook] Epoch {cur_epoch}: {self.metric}={value:.4f} < {self.threshold:.2f} "
                    f"({self._bad_epochs}/{self.patience})")
        else:
            # Reset counter on improvement beyond threshold
            if self._bad_epochs != 0 and self.verbose:
                logger.info(f"[EarlyStopHook] Reset counter as {self.metric}={value:.4f} >= {self.threshold:.2f}")
            self._bad_epochs = 0

        if self._bad_epochs >= self.patience:
            # Signal runner to stop after current epoch
            if self.verbose:
                logger.warning(
                    f"[EarlyStopHook] Trigger early stop at epoch {cur_epoch}: "
                    f"{self.metric} stayed below {self.threshold:.2f} for {self.patience} epochs.")
            # Reduce max_epochs so the loop ends naturally
            try:
                runner._max_epochs = runner.epoch + 1
            except Exception:
                # Fallback: mark a flag
                runner.message_hub.update_info('early_stop', True)
