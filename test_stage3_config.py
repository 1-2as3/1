"""
Stage3 Configuration Validation Script
Validates:
1. Config loading
2. Model building with custom modules
3. Dataset loading with ConcatDataset
4. Forward pass test
"""

import sys
sys.path.insert(0, '.')

from mmengine.config import Config
from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules
import torch

def validate_stage3_config():
    print("=" * 80)
    print("STAGE 3 CONFIGURATION VALIDATION")
    print("=" * 80)
    
    # Register all modules
    register_all_modules(init_default_scope=True)
    print("\n[OK] Registered all modules")
    
    # Load config
    config_file = 'configs/llvip/stage3_joint_multimodal.py'
    print(f"\n[1/4] Loading config: {config_file}")
    try:
        cfg = Config.fromfile(config_file)
        print(f"  [OK] Config loaded successfully")
        print(f"  - Work dir: {cfg.work_dir}")
        print(f"  - Load from: {cfg.load_from}")
        print(f"  - Max epochs: {cfg.train_cfg.max_epochs}")
    except Exception as e:
        print(f"  [X] Config loading failed: {e}")
        return False
    
    # Check model config
    print("\n[2/4] Validating model configuration")
    try:
        model_cfg = cfg.model
        print(f"  [OK] Model type: {model_cfg['type']}")
        
        # Check neck MSP
        if 'neck' in model_cfg and model_cfg['neck'].get('use_msp'):
            print(f"  [OK] Neck has MSP: {model_cfg['neck']['msp_module']}")
        else:
            print(f"  [X] WARNING: Neck MSP not configured")
        
        # Check RoI head
        roi_head = model_cfg.get('roi_head', {})
        print(f"  - RoI head type: {roi_head.get('type')}")
        print(f"  - use_macl: {roi_head.get('use_macl')}")
        print(f"  - use_msp: {roi_head.get('use_msp')}")
        print(f"  - use_dhn: {roi_head.get('use_dhn')}")
        print(f"  - lambda1 (MACL): {roi_head.get('lambda1')}")
        print(f"  - lambda2 (DHN): {roi_head.get('lambda2')}")
        print(f"  - lambda3 (Domain): {roi_head.get('lambda3')}")
        
        if 'macl_head' in roi_head:
            macl = roi_head['macl_head']
            print(f"  [OK] MACL head configured: temperature={macl.get('temperature')}, dhn_cfg={macl.get('dhn_cfg')}")
        else:
            print(f"  [X] WARNING: MACL head not configured")
            
    except Exception as e:
        print(f"  [X] Model config validation failed: {e}")
        return False
    
    # Check dataset config
    print("\n[3/4] Validating dataset configuration")
    try:
        train_data = cfg.train_dataloader.dataset
        print(f"  [OK] Train dataset type: {train_data['type']}")
        
        if train_data['type'] == 'ConcatDataset':
            datasets = train_data['datasets']
            print(f"  [OK] Number of datasets: {len(datasets)}")
            for i, ds in enumerate(datasets):
                print(f"    - Dataset {i+1}: {ds['type']}")
                print(f"      * data_root: {ds['data_root']}")
                print(f"      * ann_file: {ds['ann_file']}")
                print(f"      * pipeline: {'Configured' if 'pipeline' in ds else 'MISSING'}")
        else:
            print(f"  [X] WARNING: Expected ConcatDataset, got {train_data['type']}")
        
        val_data = cfg.val_dataloader.dataset
        print(f"  [OK] Val dataset type: {val_data['type']}")
        print(f"    - data_root: {val_data['data_root']}")
        
    except Exception as e:
        print(f"  [X] Dataset config validation failed: {e}")
        return False
    
    # Try building model
    print("\n[4/4] Building model")
    try:
        # Prepare model config for building
        model_cfg_copy = cfg.model.copy()
        model_cfg_copy['data_preprocessor'] = dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32
        )
        
        model = MODELS.build(model_cfg_copy)
        model.eval()
        print(f"  [OK] Model built successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Check for custom modules
        has_msp = False
        has_macl = False
        for name, module in model.named_modules():
            if 'MSPReweight' in str(type(module)):
                has_msp = True
                print(f"  [OK] Found MSP module: {name}")
            if 'MACLHead' in str(type(module)):
                has_macl = True
                print(f"  [OK] Found MACL module: {name}")
        
        if not has_msp:
            print(f"  [X] WARNING: MSP module not found in model")
        if not has_macl:
            print(f"  [X] WARNING: MACL module not found in model")
            
        # Quick forward pass test
        print(f"\n  Testing forward pass...")
        batch_size = 2
        dummy_input = {
            'inputs': [torch.randn(3, 640, 640) for _ in range(batch_size)],
            'data_samples': []
        }
        
        from mmdet.structures import DetDataSample
        for i in range(batch_size):
            sample = DetDataSample()
            sample.set_metainfo({
                'img_shape': (640, 640),
                'scale_factor': (1.0, 1.0),
                'ori_shape': (640, 640),
                'modality': 'infrared' if i % 2 == 0 else 'visible'
            })
            dummy_input['data_samples'].append(sample)
        
        with torch.no_grad():
            output = model.forward(**dummy_input, mode='loss')
        print(f"  [OK] Forward pass successful")
        print(f"  - Output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"  [X] Model building/forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("[SUCCESS] STAGE 3 CONFIGURATION VALIDATION PASSED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Ensure Stage2 weights exist: work_dirs/stage2_kaist_domain_ft/epoch_12.pth")
    print("2. Verify dataset paths are accessible:")
    print("   - C:/KAIST_processed/")
    print("   - C:/M3FD_processed/")
    print("3. Run training with: python tools/train.py configs/llvip/stage3_joint_multimodal.py")
    print("\nRecommended pre-training checks:")
    print("- Run deep_check.py to verify no hidden issues")
    print("- Verify backbone is frozen (if intended)")
    print("- Check GPU memory availability for batch_size=2")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    success = validate_stage3_config()
    sys.exit(0 if success else 1)
