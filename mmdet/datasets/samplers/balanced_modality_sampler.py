# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import Sampler
from typing import Iterator


class BalancedModalitySampler(Sampler):
    """Sampler that balances different modalities in each batch.
    
    This sampler ensures that each batch contains roughly equal numbers of
    samples from different modalities (e.g., visible/infrared, lwir/mwir).
    
    Args:
        dataset: The dataset to sample from.
        modality_keys (list): List of modality identifiers to balance.
            Defaults to ['visible', 'infrared', 'lwir', 'mwir'].
    """
    
    def __init__(self, dataset, modality_keys=None):
        self.dataset = dataset
        if modality_keys is None:
            modality_keys = ['visible', 'infrared', 'lwir', 'mwir']
        
        # 收集各模态的索引
        self.modality_indices = {key: [] for key in modality_keys}
        
        for i in range(len(dataset)):
            try:
                # 尝试获取数据样本的模态信息
                data = dataset[i]
                if isinstance(data, dict) and 'data_samples' in data:
                    modality = data['data_samples'].metainfo.get('modality', '')
                else:
                    modality = ''
                
                # 根据模态信息分类索引
                for key in modality_keys:
                    if key in modality.lower():
                        self.modality_indices[key].append(i)
                        break
            except Exception as e:
                print(f"[WARN] Failed to get modality for sample {i}: {e}")
                continue
        
        # 打印统计信息
        for key, indices in self.modality_indices.items():
            if len(indices) > 0:
                print(f"[BalancedModalitySampler] Found {len(indices)} samples for modality '{key}'")
    
    def __iter__(self) -> Iterator[int]:
        """Generate balanced sampling indices."""
        # 找出最小模态的样本数
        non_empty_modalities = {k: v for k, v in self.modality_indices.items() if len(v) > 0}
        
        if len(non_empty_modalities) == 0:
            return iter([])
        
        min_length = min(len(indices) for indices in non_empty_modalities.values())
        
        # 交替采样各模态
        indices = []
        modality_lists = [indices[:min_length] for indices in non_empty_modalities.values()]
        
        for sample_indices in zip(*modality_lists):
            indices.extend(sample_indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return the total number of samples that will be sampled."""
        non_empty_modalities = {k: v for k, v in self.modality_indices.items() if len(v) > 0}
        
        if len(non_empty_modalities) == 0:
            return 0
        
        min_length = min(len(indices) for indices in non_empty_modalities.values())
        return min_length * len(non_empty_modalities)
