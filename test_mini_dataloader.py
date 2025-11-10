"""
Quick test with mini dataset (100 samples)
"""
import sys
sys.path.insert(0, '.')

from mmengine.registry import DATASETS
from mmdet.utils import register_all_modules

# Register modules
print("Registering modules...")
register_all_modules(init_default_scope=True)

# Build mini dataset
print("Building KAIST mini dataset (100 samples)...")
dataset_cfg = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_subdir='Annotations',
    ann_file='C:/KAIST_processed/ImageSets/train_mini.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    return_modality_pair=True,
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(640, 640), keep_ratio=False),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'modality')
        )
    ]
)

train_dataset = DATASETS.build(dataset_cfg)
print(f"✓ Dataset built with {len(train_dataset)} samples")

# Try to load first 3 samples
print("\nTesting sample loading:")
for i in range(min(3, len(train_dataset))):
    try:
        print(f"  Loading sample {i}...", end=" ")
        sample = train_dataset[i]
        print(f"✓ Keys: {list(sample.keys())}")
        
        if 'data_samples' in sample:
            metainfo = sample['data_samples'].metainfo
            print(f"    Modality: {metainfo.get('modality', 'N/A')}")
            print(f"    Paired path: {metainfo.get('paired_path', 'N/A')[:60]}...")
            print(f"    Paired exists: {metainfo.get('paired_exists', 'N/A')}")
        
        if 'inputs' in sample:
            print(f"    Input shape: {sample['inputs'].shape}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Test dataloader
print("\n" + "="*60)
print("Testing DataLoader with batch_size=4...")
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate

try:
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=pseudo_collate
    )
    
    print("✓ DataLoader created")
    print("Fetching first batch...", end=" ")
    
    batch = next(iter(train_loader))
    print(f"✓ Success!")
    print(f"  Batch type: {type(batch)}")
    print(f"  Batch length: {len(batch)}")
    
    if isinstance(batch, dict):
        print(f"  Batch keys: {list(batch.keys())}")
        if 'inputs' in batch:
            print(f"  Inputs shape: {batch['inputs'].shape}")
    elif isinstance(batch, (list, tuple)):
        print(f"  First item keys: {list(batch[0].keys()) if hasattr(batch[0], 'keys') else 'N/A'}")
        if hasattr(batch[0], 'keys') and 'inputs' in batch[0]:
            print(f"  First item inputs shape: {batch[0]['inputs'].shape}")
    
    print("\n✅ All tests PASSED!")
    
except Exception as e:
    print(f"✗ FAILED")
    print(f"\n❌ ERROR:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
