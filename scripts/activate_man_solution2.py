"""
方案二：实现跨模态对齐模块（CrossModalAligner）

如果您需要的是跨模态特征对齐（让 visible 和 infrared 特征相似），
而非模态自适应归一化，则需要创建新的 CrossModalAligner 模块。

这个脚本：
1. 创建 CrossModalAligner 模块
2. 修改 StandardRoIHead 以支持跨模态对齐
3. 更新 Stage3 配置

注意：此方案需要修改核心代码，比方案一复杂。
"""

import os
from pathlib import Path
import shutil

def create_cross_modal_aligner():
    """创建 CrossModalAligner 模块"""
    
    aligner_path = Path(r"C:\Users\Xinyu\mmdetection\mmdet\models\macldhnmsp\cross_modal_aligner.py")
    
    print("=" * 80)
    print("CREATING CROSS-MODAL ALIGNER MODULE (方案二)")
    print("=" * 80)
    
    print(f"\n[1] Creating CrossModalAligner module...")
    
    aligner_code = '''"""
Cross-Modal Alignment Module

This module aligns features from different modalities (visible and infrared)
by minimizing the distance between their representations.

Different from ModalityAdaptiveNorm which uses separate BN for different modalities,
this module actively pushes visible and infrared features closer in the feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class CrossModalAligner(nn.Module):
    """Cross-Modal Feature Alignment Module.
    
    Aligns visible and infrared features by projecting them to a shared space
    and minimizing the distance between their distributions.
    
    Args:
        in_dim (int): Input feature dimension. Defaults to 256.
        hidden_dim (int): Hidden projection dimension. Defaults to 128.
        align_type (str): Type of alignment loss. 
            Options: 'mse', 'cosine', 'kl'. Defaults to 'mse'.
        use_projection (bool): Whether to use projection layers.
            Defaults to True.
    """
    
    def __init__(self, 
                 in_dim=256, 
                 hidden_dim=128,
                 align_type='mse',
                 use_projection=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.align_type = align_type
        self.use_projection = use_projection
        
        if use_projection:
            # Separate projections for each modality
            self.vis_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.ir_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Batch normalization for stability
            self.bn_vis = nn.BatchNorm1d(hidden_dim)
            self.bn_ir = nn.BatchNorm1d(hidden_dim)
        
        # Loss function selection
        if align_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif align_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        elif align_type == 'kl':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unknown align_type: {align_type}")
    
    def forward(self, vis_feat, ir_feat):
        """Compute cross-modal alignment loss.
        
        Args:
            vis_feat (Tensor): Visible features, shape (N, C) or (N, C, H, W).
            ir_feat (Tensor): Infrared features, shape (N, C) or (N, C, H, W).
            
        Returns:
            dict: Dictionary containing:
                - align_loss (Tensor): Alignment loss value.
                - vis_aligned (Tensor): Aligned visible features (for analysis).
                - ir_aligned (Tensor): Aligned infrared features (for analysis).
        """
        # Handle 4D feature maps (N, C, H, W) -> (N, C)
        if vis_feat.dim() == 4:
            vis_feat = F.adaptive_avg_pool2d(vis_feat, 1).flatten(1)
        if ir_feat.dim() == 4:
            ir_feat = F.adaptive_avg_pool2d(ir_feat, 1).flatten(1)
        
        # Project to shared space
        if self.use_projection:
            vis_aligned = self.bn_vis(self.vis_proj(vis_feat))
            ir_aligned = self.bn_ir(self.ir_proj(ir_feat))
        else:
            vis_aligned = vis_feat
            ir_aligned = ir_feat
        
        # Compute alignment loss
        if self.align_type == 'mse':
            # MSE between mean representations
            vis_mean = vis_aligned.mean(0)
            ir_mean = ir_aligned.mean(0)
            align_loss = self.loss_fn(vis_mean, ir_mean)
            
        elif self.align_type == 'cosine':
            # Cosine similarity loss
            target = torch.ones(vis_aligned.size(0), device=vis_aligned.device)
            align_loss = self.loss_fn(vis_aligned, ir_aligned, target)
            
        elif self.align_type == 'kl':
            # KL divergence between distributions
            vis_prob = F.softmax(vis_aligned, dim=-1)
            ir_log_prob = F.log_softmax(ir_aligned, dim=-1)
            align_loss = self.loss_fn(ir_log_prob, vis_prob)
        
        return {
            'align_loss': align_loss,
            'vis_aligned': vis_aligned,
            'ir_aligned': ir_aligned
        }
    
    def get_aligned_features(self, feat, modality='visible'):
        """Get aligned features for single modality (for inference).
        
        Args:
            feat (Tensor): Input features.
            modality (str): 'visible' or 'infrared'.
            
        Returns:
            Tensor: Aligned features.
        """
        if not self.use_projection:
            return feat
        
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        
        if modality == 'visible':
            return self.bn_vis(self.vis_proj(feat))
        else:
            return self.bn_ir(self.ir_proj(feat))
'''
    
    # 写入文件
    with open(aligner_path, 'w', encoding='utf-8') as f:
        f.write(aligner_code)
    
    print(f"  [OK] Created: {aligner_path}")
    print(f"  Module: CrossModalAligner")
    print(f"  Features:")
    print(f"    - Align vis/ir features in shared space")
    print(f"    - Support MSE/Cosine/KL alignment")
    print(f"    - Optional projection layers")
    
    # 更新 __init__.py
    init_path = aligner_path.parent / "__init__.py"
    
    with open(init_path, 'r', encoding='utf-8') as f:
        init_content = f.read()
    
    if 'CrossModalAligner' not in init_content:
        # 添加导入
        new_import = "from .cross_modal_aligner import CrossModalAligner\n"
        
        # 在第一个 from 语句后插入
        lines = init_content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from .'):
                insert_idx = i + 1
        
        lines.insert(insert_idx, new_import.rstrip())
        
        # 更新 __all__
        for i, line in enumerate(lines):
            if '__all__' in line:
                # 找到 __all__ 列表
                all_line = line
                if 'CrossModalAligner' not in all_line:
                    # 在列表最后添加
                    all_line = all_line.rstrip(']').rstrip()
                    if not all_line.endswith(','):
                        all_line += ','
                    all_line += " 'CrossModalAligner']"
                    lines[i] = all_line
                break
        
        new_init_content = '\n'.join(lines)
        
        # 备份
        backup_path = init_path.with_suffix('.py.backup_cma')
        shutil.copy(init_path, backup_path)
        
        # 写入
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(new_init_content)
        
        print(f"\n[2] Updated __init__.py")
        print(f"  Added CrossModalAligner to exports")
        print(f"  Backup: {backup_path.name}")
    else:
        print(f"\n[2] __init__.py already exports CrossModalAligner")
    
    return True


def show_roi_head_integration_guide():
    """显示如何集成到 StandardRoIHead 的指南"""
    
    print(f"\n[3] StandardRoIHead Integration Guide")
    print(f"\n  To integrate CrossModalAligner into StandardRoIHead:")
    print(f"\n  Step 1: Add parameters to __init__:")
    print(f"  ```python")
    print(f"  def __init__(self, ...")
    print(f"               use_cross_modal_align: bool = False,")
    print(f"               cross_modal_aligner: ConfigType = None,")
    print(f"               lambda4: float = 0.2,  # Alignment loss weight")
    print(f"               **kwargs):")
    print(f"      self.use_cross_modal_align = use_cross_modal_align")
    print(f"      self.lambda4 = lambda4")
    print(f"      ")
    print(f"      if use_cross_modal_align:")
    print(f"          self.cross_modal_aligner = MODELS.build(cross_modal_aligner)")
    print(f"  ```")
    print(f"\n  Step 2: Add to loss() method:")
    print(f"  ```python")
    print(f"  if self.use_cross_modal_align and has_paired_modality:")
    print(f"      # Extract vis/ir features")
    print(f"      vis_feats = x[vis_indices]")
    print(f"      ir_feats = x[ir_indices]")
    print(f"      ")
    print(f"      # Compute alignment loss")
    print(f"      align_result = self.cross_modal_aligner(vis_feats, ir_feats)")
    print(f"      losses['loss_cross_modal_align'] = align_result['align_loss'] * self.lambda4")
    print(f"  ```")
    print(f"\n  Step 3: Update Stage3 config:")
    print(f"  ```python")
    print(f"  roi_head=dict(")
    print(f"      type='StandardRoIHead',")
    print(f"      # ... existing params")
    print(f"      use_cross_modal_align=True,")
    print(f"      cross_modal_aligner=dict(")
    print(f"          type='CrossModalAligner',")
    print(f"          in_dim=256,")
    print(f"          hidden_dim=128,")
    print(f"          align_type='mse'")
    print(f"      ),")
    print(f"      lambda4=0.2,")
    print(f"  )")
    print(f"  ```")


def main():
    print("This script creates the CrossModalAligner module")
    print("for cross-modal feature alignment.")
    print()
    print("NOTE: This requires manual integration into StandardRoIHead.")
    print("      Detailed instructions will be provided after creation.")
    print()
    
    response = input("Continue? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    # 创建模块
    success = create_cross_modal_aligner()
    
    if success:
        # 显示集成指南
        show_roi_head_integration_guide()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] CrossModalAligner module created!")
        print("=" * 80)
        print("\nFiles created/modified:")
        print("  - mmdet/models/macldhnmsp/cross_modal_aligner.py (NEW)")
        print("  - mmdet/models/macldhnmsp/__init__.py (UPDATED)")
        print("\nNext steps:")
        print("  1. Review the integration guide above")
        print("  2. Modify StandardRoIHead to add cross_modal_align support")
        print("  3. Update Stage3 config with use_cross_modal_align=True")
        print("  4. Test with: python test_stage3_config.py")
        print("=" * 80)


if __name__ == "__main__":
    main()
