"""Minimal compatible PackDetInputs for environments missing the official one.

This implementation is intentionally lightweight to enable dataset building
and basic pipeline execution in research/debug scenarios.
"""
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any

from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


@TRANSFORMS.register_module(name='PackDetInputs')
class PackDetInputs:
    def __init__(self, meta_keys: Optional[Sequence[str]] = None):
        self.meta_keys = tuple(meta_keys) if meta_keys is not None else None

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Image tensor
        img = results.get('img')

        # Prepare data sample with gt instances when available
        data_sample = DetDataSample()
        inst = InstanceData()
        if 'gt_bboxes' in results and results['gt_bboxes'] is not None:
            inst.bboxes = results['gt_bboxes']
        if 'gt_labels' in results and results['gt_labels'] is not None:
            inst.labels = results['gt_labels']
        if len(inst.__dict__) > 0:
            data_sample.gt_instances = inst

        # Meta info
        meta = {}
        if self.meta_keys is not None:
            for k in self.meta_keys:
                if k in results:
                    meta[k] = results[k]
        if meta:
            data_sample.set_metainfo(meta)

        return dict(inputs=img, data_samples=data_sample)
