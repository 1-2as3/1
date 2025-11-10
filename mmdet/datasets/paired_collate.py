"""Paired modality collate function.

将单个样本 (含 data_samples 和 infrared_img 属性) 列表组装成 batch。
如果样本中包含 `data_samples.infrared_img`，则生成 `inputs` 为可见光 batch，
并在每个 data_sample 上保留 `infrared_img` 以便 detector 在 loss 阶段构造 paired dict。

返回结构:
	{
		'inputs': Tensor (N,C,H,W) 可见光;
		'data_samples': List[DetDataSample] (每个可能带 infrared_img Tensor)
	}
"""

from __future__ import annotations

from typing import List
import torch

def paired_modality_collate(batch: List[dict]):
	vis_imgs = []
	data_samples = []
	for item in batch:
		# 标准 pack 后 item 结构: {'inputs': Tensor, 'data_samples': DetDataSample}
		if 'inputs' not in item or 'data_samples' not in item:
			raise ValueError('[paired_modality_collate] invalid item keys: ' + str(item.keys()))
		vis_imgs.append(item['inputs'])
		data_samples.append(item['data_samples'])
	# stack visible images
	inputs = torch.stack(vis_imgs, dim=0)
	return {'inputs': inputs, 'data_samples': data_samples}

__all__ = ['paired_modality_collate']
