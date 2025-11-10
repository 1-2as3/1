"""Debug KAIST dataset loading"""
import sys
sys.path.insert(0, 'c:\\Users\\Xinyu\\mmdetection')

from mmdet.datasets import KAISTDataset
from mmdet.registry import DATASETS

# Test configuration
cfg = dict(
    type='KAISTDataset',
    data_root='C:/KAIST_processed/',
    ann_subdir='Annotations',
    ann_file='C:/KAIST_processed/ImageSets/train.txt',
    data_prefix=dict(sub_data_root='C:/KAIST_processed'),
    return_modality_pair=False,
    pipeline=[],  # Empty pipeline for debugging
    filter_cfg=None,  # Disable filtering
    test_mode=False,
    serialize_data=False,  # Disable serialization
)

print("Building dataset...")
dataset = DATASETS.build(cfg)

print(f"\nDataset length: {len(dataset.data_list)}")

if len(dataset.data_list) > 0:
    print(f"\nFirst sample:")
    print(dataset.data_list[0])
else:
    print("\n[ERROR] No samples loaded!")
    
    # Check raw img list
    print(f"\nChecking raw image list...")
    img_ids = dataset.img_ids if hasattr(dataset, 'img_ids') else []
    print(f"Raw img_ids count: {len(img_ids)}")
    if len(img_ids) > 0:
        print(f"First 5 img_ids: {img_ids[:5]}")
