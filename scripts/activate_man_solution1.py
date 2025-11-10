"""
方案一：激活现有 ModalityAdaptiveNorm（推荐方案）

这个脚本修改 Stage3 配置以使用模态自适应 BN。
模态自适应 BN 在 backbone 层面为不同模态使用独立的归一化层。

优点：
- 模块已实现，代码质量高
- 无需修改 StandardRoIHead
- 在特征提取阶段就进行模态自适应
"""

import os
from pathlib import Path

def activate_modality_adaptive_norm():
    """修改 Stage3 配置以激活 ModalityAdaptiveResNet"""
    
    stage3_config = Path(r"C:\Users\Xinyu\mmdetection\configs\llvip\stage3_joint_multimodal.py")
    
    if not stage3_config.exists():
        print(f"[ERROR] Config file not found: {stage3_config}")
        return False
    
    print("=" * 80)
    print("ACTIVATING MODALITY-ADAPTIVE NORMALIZATION (方案一)")
    print("=" * 80)
    
    # 读取现有配置
    with open(stage3_config, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n[1] Reading current config...")
    print(f"  File: {stage3_config.name}")
    print(f"  Size: {len(content)} chars")
    
    # 检查是否已包含 ModalityAdaptiveResNet
    if 'ModalityAdaptiveResNet' in content:
        print(f"\n[INFO] ModalityAdaptiveResNet already configured!")
        return True
    
    # 查找 model = dict( 块
    import re
    
    # 查找 model dict 的起始位置
    model_match = re.search(r'model\s*=\s*dict\s*\(', content)
    if not model_match:
        print(f"[ERROR] Cannot find 'model = dict(' in config")
        return False
    
    model_start = model_match.end()
    
    # 在 type='FasterRCNN' 后插入 backbone 配置
    type_match = re.search(r"type\s*=\s*['\"]FasterRCNN['\"]", content[model_start:])
    if not type_match:
        print(f"[ERROR] Cannot find type='FasterRCNN' in model dict")
        return False
    
    insert_pos = model_start + type_match.end()
    
    # 检查是否已有 backbone 配置
    if re.search(r'backbone\s*=\s*dict', content[insert_pos:insert_pos+1000]):
        print(f"[WARNING] Backbone already configured, manual review needed")
        return False
    
    # 准备插入的 backbone 配置
    backbone_config = """,
    # Modality-Adaptive ResNet Backbone
    backbone=dict(
        type='ModalityAdaptiveResNet',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        use_modality_bn=True,
        modality_bn_layers=[0, 1]  # Apply MAN to first two layers
    )"""
    
    # 插入配置
    new_content = content[:insert_pos] + backbone_config + content[insert_pos:]
    
    # 备份原配置
    backup_path = stage3_config.with_suffix('.py.backup_man')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[2] Backup created: {backup_path.name}")
    
    # 写入新配置
    with open(stage3_config, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n[3] Updated config with ModalityAdaptiveResNet")
    print(f"  Added backbone configuration:")
    print(f"    - Type: ModalityAdaptiveResNet")
    print(f"    - Base backbone: ResNet50")
    print(f"    - MAN layers: [0, 1] (stem + layer1)")
    
    print(f"\n[SUCCESS] Configuration updated!")
    print(f"\nWhat was changed:")
    print(f"  - Added ModalityAdaptiveResNet as backbone")
    print(f"  - Modality-adaptive BN on first two layers")
    print(f"  - Separate BN statistics for visible/infrared")
    
    print(f"\nNext steps:")
    print(f"  1. Verify config loads: python -c \"from mmengine.config import Config; Config.fromfile('{stage3_config}')\"")
    print(f"  2. Test model building: python test_stage3_config.py")
    print(f"  3. Start training: python tools/train.py {stage3_config}")
    
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    print("This script will modify Stage3 config to use ModalityAdaptiveResNet")
    print("A backup will be created automatically.")
    print()
    
    response = input("Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = activate_modality_adaptive_norm()
        exit(0 if success else 1)
    else:
        print("Operation cancelled.")
        exit(0)
