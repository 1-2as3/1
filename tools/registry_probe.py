"""Probe registry presence for key symbols used in pipelines and datasets."""
from mmdet.utils import register_all_modules
from mmengine.registry import TRANSFORMS as ME_TRANSFORMS
from mmdet.registry import DATASETS, TRANSFORMS as MMDET_TRANSFORMS

def main():
    register_all_modules(init_default_scope=True)
    keys_t = list(ME_TRANSFORMS.module_dict.keys())[:50]
    print('[INFO] mmengine.TRANSFORMS has', len(ME_TRANSFORMS.module_dict), 'entries')
    print('[INFO] Example mmengine.TRANSFORMS keys:', keys_t)
    for k in ['PackDetInputs', 'LoadImageFromFile', 'LoadAnnotations', 'Resize', 'RandomFlip']:
        print(f"- {k} in mmengine.TRANSFORMS:", k in ME_TRANSFORMS.module_dict)

    keys_t2 = list(MMDET_TRANSFORMS.module_dict.keys())[:50]
    print('[INFO] mmdet.TRANSFORMS has', len(MMDET_TRANSFORMS.module_dict), 'entries')
    print('[INFO] Example mmdet.TRANSFORMS keys:', keys_t2)
    for k in ['PackDetInputs', 'LoadImageFromFile', 'LoadAnnotations', 'Resize', 'RandomFlip']:
        print(f"- {k} in mmdet.TRANSFORMS:", k in MMDET_TRANSFORMS.module_dict)

    ds_keys = list(DATASETS.module_dict.keys())[:50]
    print('[INFO] DATASETS has', len(DATASETS.module_dict), 'entries')
    print('[INFO] Example DATASETS keys:', ds_keys)
    for k in ['KAISTDataset']:
        print(f"- {k} in DATASETS:", k in DATASETS.module_dict)

if __name__ == '__main__':
    main()
