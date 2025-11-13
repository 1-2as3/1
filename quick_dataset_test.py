"""Quick test of KAISTDataset instantiation"""
from mmdet.datasets import KAISTDataset
from mmdet.utils import register_all_modules
from mmdet.datasets.transforms import LoadImageFromFile, LoadAnnotations, Resize, RandomFlip, PackDetInputs

register_all_modules()

print("[Test] Creating KAISTDataset with return_modality_pair=True...")

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

dataset = KAISTDataset(
    data_root='C:/KAIST_processed/',
    ann_file='ImageSets/train.txt',
    data_prefix=dict(sub_data_root='./'),
    metainfo=dict(classes=('person',)),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=pipeline,
    return_modality_pair=True,
)

print(f"[OK] Dataset created")
print(f"     - type: {type(dataset)}")
print(f"     - return_modality_pair: {dataset.return_modality_pair}")
print(f"     - len: {len(dataset)}")

print("\n[Test] Loading first sample...")
data = dataset[0]

print(f"[OK] Data loaded")
print(f"     - keys: {data.keys()}")

if 'data_samples' in data:
    has_infrared = hasattr(data['data_samples'], 'infrared_img')
    print(f"     - has infrared_img: {has_infrared}")
    if has_infrared:
        print(f"     - infrared_img shape: {data['data_samples'].infrared_img.shape}")
    else:
        print("     - [PROBLEM] infrared_img NOT FOUND in data_samples!")
else:
    print("     - [PROBLEM] data_samples key not in data!")

print("\n[Test Complete]")
