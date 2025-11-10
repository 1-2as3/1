"""Domain Weight Warmup Hook

按 epoch 线性/自定义策略对 `roi_head.macl_head.domain_weight` 进行热身，
用于避免域对齐损失在训练初期过强干扰主任务收敛。

配置示例 (加入到 custom_hooks):
	dict(
		type='DomainWeightWarmupHook',
		attr_path='roi_head.macl_head.domain_weight',
		start=0.0,
		target=0.1,
		warmup_epochs=2,
		mode='linear',
		verbose=True,
	)

参数:
	attr_path (str): 从 runner.model (DDP unwrap 后) 根开始的属性路径，用 '.' 分隔。
	start (float): 初始权重值。若设置且与当前不同，会在 before_train_epoch 强制写入。
	target (float): 热身结束时的目标值。
	warmup_epochs (int): 热身持续 epoch 数 (>=1)。
	mode (str): 'linear' 或 'exp'；exp 为指数缓升。
	verbose (bool): 输出日志。
	clamp_min/max (Optional[float]): 额外的数值范围限制。

行为:
	- 每个 epoch 开始 (before_train_epoch) 根据当前 epoch 计算新的权重并写回。
	- 过了 warmup_epochs 后保持 target 不再变化。
	- 支持空跑 (warmup_epochs<=0) 直接设定 target。
"""

from __future__ import annotations

from typing import Optional
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _resolve_attr(root, path: str):
	cur = root
	for part in path.split('.'):
		if not hasattr(cur, part):
			return None
		cur = getattr(cur, part)
	return cur


def _set_attr(root, path: str, value):
	parts = path.split('.')
	cur = root
	for p in parts[:-1]:
		if not hasattr(cur, p):
			return False
		cur = getattr(cur, p)
	last = parts[-1]
	if not hasattr(cur, last):
		return False
	setattr(cur, last, value)
	return True


@HOOKS.register_module()
class DomainWeightWarmupHook(Hook):
	priority = 'NORMAL'

	def __init__(
		self,
		attr_path: str = 'roi_head.macl_head.domain_weight',
		start: float = 0.0,
		target: float = 0.1,
		warmup_epochs: int = 2,
		mode: str = 'linear',  # 'linear' | 'exp'
		verbose: bool = True,
		clamp_min: Optional[float] = None,
		clamp_max: Optional[float] = None,
	) -> None:
		self.attr_path = attr_path
		self.start = float(start)
		self.target = float(target)
		self.warmup_epochs = int(warmup_epochs)
		self.mode = mode
		self.verbose = verbose
		self.clamp_min = clamp_min
		self.clamp_max = clamp_max
		if self.warmup_epochs < 0:
			self.warmup_epochs = 0

	def before_train(self, runner):  # 在整体训练开始时记录初始值
		model = runner.model.module if hasattr(runner.model, 'module') else runner.model
		current = _resolve_attr(model, self.attr_path)
		if current is None:
			runner.logger.warning(f'[DomainWeightWarmupHook] attr_path not found: {self.attr_path}')
			return
		if self.verbose:
			runner.logger.info(f'[DomainWeightWarmupHook] initial value={current}')

	def before_train_epoch(self, runner):
		model = runner.model.module if hasattr(runner.model, 'module') else runner.model
		epoch = runner.epoch  # 从0开始
		# 计算权重
		if self.warmup_epochs <= 0:
			new_w = self.target
		else:
			if epoch >= self.warmup_epochs:
				new_w = self.target
			else:
				progress = (epoch + 1) / float(self.warmup_epochs)  # 用 (epoch+1) 使第1个epoch后已有正值
				if self.mode == 'exp':
					# 指数平滑：start + (target-start)*(1 - exp(-5*progress)) / (1 - exp(-5))
					import math
					denom = 1 - math.exp(-5)
					num = 1 - math.exp(-5 * progress)
					frac = num / denom if denom != 0 else progress
					new_w = self.start + (self.target - self.start) * frac
				else:  # linear
					new_w = self.start + (self.target - self.start) * progress

		# Clamp
		if self.clamp_min is not None:
			new_w = max(new_w, self.clamp_min)
		if self.clamp_max is not None:
			new_w = min(new_w, self.clamp_max)

		ok = _set_attr(model, self.attr_path, float(new_w))
		if not ok:
			runner.logger.warning(f'[DomainWeightWarmupHook] Failed to set {self.attr_path}')
			return
		if self.verbose:
			runner.logger.info(
				f'[DomainWeightWarmupHook] epoch={epoch} set {self.attr_path}={new_w:.6f} (target={self.target}, warmup_epochs={self.warmup_epochs})'
			)

	def after_train_epoch(self, runner):
		# 记录最终值（便于在日志中确认）
		model = runner.model.module if hasattr(runner.model, 'module') else runner.model
		current = _resolve_attr(model, self.attr_path)
		if self.verbose and current is not None:
			runner.logger.info(f'[DomainWeightWarmupHook] confirm {self.attr_path}={current:.6f} after epoch {runner.epoch}')

