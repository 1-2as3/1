"""
Test script to verify dataloader batch collation
"""
import sys
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.registry import DATASETS, TRANSFORMS
from mmdet.utils import register_all_modules

# Register all modules
print("Registering modules...")
register_all_modules(init_default_scope=True)
print("✓ Modules registered")

# Load config
print("Loading config...")
cfg = Config.fromfile('configs/llvip/stage2_llvip_kaist_finetune_sanity.py')
print("✓ Config loaded")

# Build dataset
print("Building train dataset...")
print(f"  Dataset config: {cfg.train_dataloader.dataset}")
train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f"✓ Dataset built")
print(f"Dataset size: {len(train_dataset)}")

# Try to load a few samples
print("\nTesting sample loading:")
for i in range(3):
    try:
        sample = train_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Keys: {sample.keys() if hasattr(sample, 'keys') else type(sample)}")
        if hasattr(sample, 'keys'):
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"    {key}: len={len(value)}, type={type(value)}")
                else:
                    print(f"    {key}: {type(value)}")
    except Exception as e:
        print(f"  Error loading sample {i}: {e}")
        import traceback
        traceback.print_exc()

# Now test dataloader with batch collation
print("\n" + "="*60)
print("Testing dataloader with batch_size=4...")
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
    
    print(f"DataLoader created successfully")
    print("Attempting to fetch first batch...")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Type: {type(batch)}")
        print(f"  Len: {len(batch)}")
        
        # Examine structure
        if isinstance(batch, dict):
            print("  Batch is dict with keys:", batch.keys())
        elif isinstance(batch, (list, tuple)):
            print(f"  Batch is {type(batch).__name__} with {len(batch)} items")
            for i, item in enumerate(batch[:2]):  # Check first 2 items
                print(f"    Item {i}: {type(item)}")
                if hasattr(item, 'keys'):
                    for key in item.keys():
                        val = item[key]
                        if hasattr(val, 'shape'):
                            print(f"      {key}: shape={val.shape}")
                        else:
                            print(f"      {key}: {type(val)}")
        
        # Only test first batch
        break
        
except Exception as e:
    print(f"\n❌ ERROR during batch loading:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")
