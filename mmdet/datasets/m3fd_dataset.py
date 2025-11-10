# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.transforms import LoadAnnotations, LoadImageFromFile


@DATASETS.register_module()
class M3FDDataset(BaseDetDataset):
    """M3FD dataset adapter inheriting BaseDetDataset.

    This class converts YOLO-style annotations into the internal mmdet
    format via overriding `parse_data_info` while keeping the dataset
    interface consistent with other datasets (e.g., LLVIPDataset).
    
    Person-only detection (unified with LLVIP/KAIST).
    """
    METAINFO = {'classes': ('person',), 'dataset_type': 'M3FD', 'palette': [(220, 20, 60)]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info):
        """Parse raw YOLO-style line/record into mmdet data_info.

        Expected `raw_data_info` format depends on how the dataset is
        constructed (e.g., a dict or a string path). We call the parent
        implementation first and then add `modality`.
        """
        data_info = super().parse_data_info(raw_data_info)
        img_path = data_info.get('img_path', '')
        if 'visible' in img_path.replace('\\', '/').lower():
            modality = 'visible'
        elif 'infrared' in img_path.replace('\\', '/').lower():
            modality = 'infrared'
        else:
            modality = 'unknown'
        data_info['modality'] = modality
        return data_info

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # Ensure metainfo.modalitiy is present on DataSample for downstream use
        if isinstance(data, dict) and 'data_samples' in data:
            meta = data['data_samples'].metainfo
            # copy modality from top-level data if present
            if 'modality' in data:
                meta['modality'] = data.get('modality', 'unknown')
            else:
                # parent may have set modality inside data_samples already
                meta.setdefault('modality', meta.get('modality', 'unknown'))
        
        # 路径与标注检查
        if not data or data is None:
            raise RuntimeError(f"[M3FDDataset] Empty sample at index {idx}. Check annotation consistency.")
        if not hasattr(data['data_samples'], 'gt_instances'):
            print(f"[WARN] Missing gt_instances for sample {idx}. Verify label file formatting.")
        
        return data