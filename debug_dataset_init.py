"""
Debug script to check if KAISTDataset is initialized with correct parameters.
"""
from mmengine.config import Config
from mmengine.registry import DATASETS

# Load config
cfg = Config.fromfile('configs/llvip/stage2_2_planC_dualmodality_macl_clean.py')

print("="*60)
print("[DEBUG] Testing Dataset Initialization")
print("="*60)

print("\n[1] Config train_dataloader.dataset:")
print(cfg.train_dataloader.dataset)

print("\n[2] Building dataset from config...")
from mmdet.datasets import KAISTDataset
from mmdet.utils import register_all_modules

register_all_modules()

# Build dataset
dataset_cfg = cfg.train_dataloader.dataset
print(f"\n[3] Dataset config keys: {dataset_cfg.keys()}")
print(f"    - type: {dataset_cfg.type}")
print(f"    - return_modality_pair: {dataset_cfg.get('return_modality_pair', 'NOT FOUND')}")

# Build using registry
dataset = DATASETS.build(dataset_cfg)

print(f"\n[4] Dataset instance created:")
print(f"    - Type: {type(dataset)}")
print(f"    - Has return_modality_pair attr: {hasattr(dataset, 'return_modality_pair')}")
if hasattr(dataset, 'return_modality_pair'):
    print(f"    - return_modality_pair value: {dataset.return_modality_pair}")

print("\n[5] Testing __getitem__(0)...")
try:
    data = dataset[0]
    print(f"    - data keys: {data.keys() if isinstance(data, dict) else 'NOT A DICT'}")
    if 'data_samples' in data:
        has_infrared = hasattr(data['data_samples'], 'infrared_img')
        print(f"    - data_samples has infrared_img: {has_infrared}")
        if has_infrared:
            print(f"    - infrared_img shape: {data['data_samples'].infrared_img.shape}")
        else:
            print("    - [ERROR] infrared_img NOT FOUND!")
except Exception as e:
    print(f"    - [ERROR] {e}")

print("\n" + "="*60)
print("[DEBUG] Test Complete")
print("="*60)
