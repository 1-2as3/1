#!/usr/bin/env python
"""
Quick smoke test for dual-modality data loading on Linux.
This script verifies that:
1. KAISTDataset with return_modality_pair=True works
2. Both visible and infrared images are loaded correctly
3. Data shapes match expectations
"""

import sys
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS

def test_dual_modality_loading(config_path):
    """Test dual-modality data loading."""
    print("=" * 60)
    print("Plan C Dual-Modality Smoke Test")
    print("=" * 60)
    
    # Load config
    print(f"\n1. Loading config: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Build dataset
    print("\n2. Building dataset...")
    dataset_cfg = cfg.train_dataloader.dataset
    print(f"   - Dataset type: {dataset_cfg.type}")
    print(f"   - Data root: {dataset_cfg.data_root}")
    print(f"   - return_modality_pair: {dataset_cfg.get('return_modality_pair', False)}")
    
    dataset = DATASETS.build(dataset_cfg)
    print(f"   - Dataset length: {len(dataset)}")
    
    # Test first 5 samples
    print("\n3. Testing first 5 samples...")
    for idx in range(min(5, len(dataset))):
        data = dataset[idx]
        
        # Check structure
        assert 'inputs' in data, f"Sample {idx}: Missing 'inputs' key"
        assert 'data_samples' in data, f"Sample {idx}: Missing 'data_samples' key"
        
        # Check infrared data
        data_sample = data['data_samples']
        has_infrared = hasattr(data_sample, 'infrared_img') and data_sample.infrared_img is not None
        
        if has_infrared:
            infrared_shape = data_sample.infrared_img.shape
            visible_shape = data['inputs'].shape
            print(f"   [Sample {idx}] ✓ Dual-modality OK")
            print(f"      - Visible shape: {visible_shape}")
            print(f"      - Infrared shape: {infrared_shape}")
            assert visible_shape == infrared_shape, f"Shape mismatch! {visible_shape} != {infrared_shape}"
        else:
            print(f"   [Sample {idx}] ✗ Missing infrared_img!")
            return False
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Simple collate function for testing."""
        return {
            'inputs': [item['inputs'] for item in batch],
            'data_samples': [item['data_samples'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,  # Linux can use multiple workers
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"   - Batch size: {len(batch['inputs'])}")
    print(f"   - Batch inputs shape: {[img.shape for img in batch['inputs']]}")
    
    # Check all samples have infrared
    all_have_infrared = all(
        hasattr(ds, 'infrared_img') and ds.infrared_img is not None
        for ds in batch['data_samples']
    )
    
    if all_have_infrared:
        print(f"   - All samples in batch have infrared: ✓")
        infrared_shapes = [ds.infrared_img.shape for ds in batch['data_samples']]
        print(f"   - Batch infrared shapes: {infrared_shapes}")
    else:
        print(f"   - Some samples missing infrared: ✗")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! Dual-modality loading works correctly.")
    print("=" * 60)
    return True

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config_planC_linux.py'
    
    try:
        success = test_dual_modality_loading(config_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
