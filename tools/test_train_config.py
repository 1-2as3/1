"""Test loading dataset from training config"""
import sys
sys.path.insert(0, 'c:\\Users\\Xinyu\\mmdetection')

from mmengine.config import Config
from mmdet.registry import DATASETS

# Load training configuration
cfg = Config.fromfile('configs/llvip/stage2_llvip_kaist_finetune_sanity.py')

print("\n=== Testing Train Dataloader ===")
print(f"Dataset config: {cfg.train_dataloader.dataset.type}")
print(f"Ann file: {cfg.train_dataloader.dataset.ann_file}")
print(f"Return modality pair: {cfg.train_dataloader.dataset.get('return_modality_pair', False)}")

print("\nBuilding train dataset...")
train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f"Train dataset length: {len(train_dataset.data_list)}")

print("\n=== Testing Val Dataloader ===")
print(f"Dataset config: {cfg.val_dataloader.dataset.type}")
print(f"Ann file: {cfg.val_dataloader.dataset.ann_file}")

print("\nBuilding val dataset...")
val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
print(f"Val dataset length: {len(val_dataset.data_list)}")
