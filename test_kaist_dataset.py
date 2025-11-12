"""Test KAIST dataset loading to isolate the issue"""
import sys
sys.path.insert(0, 'C:/Users/Xinyu/mmdetection')

# Import mmdet to register all transforms
import mmdet

from mmdet.datasets import KAISTDataset

# Minimal config for testing
cfg = dict(
    data_root='C:/KAIST_processed/',
    data_prefix=dict(sub_data_root='C:/KAIST_processed/'),
    ann_file='ImageSets/train.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(640, 512), keep_ratio=True),
        dict(type='PackDetInputs')
    ],
    test_mode=False,
    return_modality_pair=False  # Test single modality first
)

print("Creating dataset...")
try:
    dataset = KAISTDataset(**cfg)
    print(f"✅ Dataset created successfully! Total samples: {len(dataset)}")
    
    print("\nTesting first sample loading...")
    sample = dataset[0]
    print(f"✅ First sample loaded successfully!")
    print(f"   Keys: {sample.keys()}")
    
    if 'inputs' in sample:
        print(f"   Image shape: {sample['inputs'].shape}")
    
    print("\n🎉 All tests passed! Dataset loading works correctly.")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
