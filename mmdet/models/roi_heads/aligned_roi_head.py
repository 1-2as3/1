"""AlignedRoIHead: Extends StandardRoIHead to integrate DomainAligner.

This class is kept minimal: it calls parent loss(), then optionally applies
DomainAligner (e.g., MMDLoss) on selected FPN level features with modality labels.
"""
from __future__ import annotations

from typing import Tuple, List
import torch
from torch import Tensor

from mmdet.registry import MODELS
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class AlignedRoIHead(StandardRoIHead):
    def __init__(self,
                 use_domain_aligner: bool = False,
                 domain_aligner=None,
                 lambda_domain: float = 0.1,
                 align_feat_level: str = 'fpn_p3',
                 **kwargs):
        # 将 use_domain_aligner 映射为父类 use_domain_alignment 使其打印状态一致
        kwargs.setdefault('use_domain_alignment', use_domain_aligner)
        self.use_domain_aligner = use_domain_aligner
        self.lambda_domain = lambda_domain
        self.align_feat_level = align_feat_level
        self.domain_aligner_cfg = domain_aligner
        super().__init__(**kwargs)

        if self.use_domain_aligner and isinstance(self.domain_aligner_cfg, dict):
            try:
                self.domain_aligner = MODELS.build(self.domain_aligner_cfg)
                print('[AlignedRoIHead] DomainAligner built.')
            except Exception as e:
                print(f'[AlignedRoIHead][WARN] DomainAligner build failed: {e}')
                self.use_domain_aligner = False

    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples: List):
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        if not self.use_domain_aligner:
            return losses
        try:
            # x here may be list/tuple of feature maps; wrap into dict style for DomainAligner reuse
            if isinstance(x, (list, tuple)):
                feats_dict = {f'fpn_p{i+2}': fm for i, fm in enumerate(x)}
            elif isinstance(x, dict):
                feats_dict = x
            else:
                return losses
            
            # modality labels from samples
            mods = []
            for sample in batch_data_samples:
                mods.append(0 if sample.metainfo.get('modality','visible') in ['visible','vis'] else 1)
            
            # Safely create modality tensor from any feature tensor (avoid calling new_tensor on tuple/list)
            ref_tensor = None
            if isinstance(x, (list, tuple)):
                for item in x:
                    if torch.is_tensor(item):
                        ref_tensor = item
                        break
            elif isinstance(x, dict):
                for v in x.values():
                    if torch.is_tensor(v):
                        ref_tensor = v
                        break
            elif torch.is_tensor(x):
                ref_tensor = x
            
            if ref_tensor is not None:
                modality_tensor = ref_tensor.new_tensor(mods, dtype=torch.long)
            else:
                # Fallback: create on CPU and let DomainAligner handle device
                modality_tensor = torch.tensor(mods, dtype=torch.long)
            
            if hasattr(self, 'domain_aligner'):
                out = self.domain_aligner(feats_dict, modality_tensor)
                if isinstance(out, dict) and 'loss_domain' in out:
                    losses['loss_domain'] = out['loss_domain'] * self.lambda_domain
        except Exception as e:
            print(f'[AlignedRoIHead][WARN] Domain alignment failed: {e}')
        return losses
