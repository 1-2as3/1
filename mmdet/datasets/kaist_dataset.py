# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from mmdet.datasets.voc import VOCDataset
import os.path as osp
from mmcv import imread
import cv2
import numpy as np


@DATASETS.register_module()
class KAISTDataset(VOCDataset):
    """KAIST Dataset for thermal-visible person detection.
    
    Supports both single-modality and paired-modality loading:
    - Single mode: Returns visible OR infrared based on sample ID
    - Paired mode: Returns both visible + infrared for the same scene
    
    Directory structure:
        C:/KAIST_processed/
        ├── Annotations/          # VOC XML files
        ├── visible/              # Visible images
        ├── infrared/             # Thermal images
        └── ImageSets/
            ├── train.txt         # Sample IDs (e.g., set00_V000_visible_I01216)
            ├── val.txt
            └── test.txt
    
    Args:
        return_modality_pair (bool): If True, __getitem__ returns both visible 
            and infrared images. If False, returns single modality based on ID.
            Default: False.
        **kwargs: Passed to parent VOCDataset class.
    """
    METAINFO = {
        'classes': ('person',),
        'dataset_type': 'KAIST',
        # Person-only detection (unified with LLVIP)
        'palette': [(220, 20, 60)]  # person in red
    }

    def __init__(self, return_modality_pair=False, **kwargs):
        """Initialize KAIST dataset.
        
        Args:
            return_modality_pair (bool): Enable paired modality loading.
            **kwargs: Passed to VOCDataset.
        """
        self.return_modality_pair = return_modality_pair
        super().__init__(**kwargs)

    def parse_data_info(self, img_info: dict):
        """Parse single sample's data info from VOC XML.
        
        Dynamically resolves visible/infrared subdirectory based on sample ID.
        
        Sample ID examples:
            - set00_V000_visible_I01216 → visible/
            - set00_V000_lwir_I01216 → infrared/
        
        Returns:
            dict: Contains img_path, xml_path, instances, etc.
        """
        data_info = super().parse_data_info(img_info)
        img_id = data_info.get('img_id', '')
        lower = img_id.lower()
        
        # Dynamically select subdirectory based on modality keyword
        if 'visible' in lower:
            subdir = 'visible'
        elif any(k in lower for k in ['lwir', 'infrared', 'thermal']):
            subdir = 'infrared'
        else:
            # Fallback to visible if no keyword found
            subdir = 'visible'

        # Override img_path to point to actual image location
        data_info['img_path'] = osp.join(self.sub_data_root, subdir, f'{img_id}.jpg')
        
        # Store modality and base ID for potential paired loading
        data_info['modality'] = 'visible' if subdir == 'visible' else 'infrared'
        data_info['base_id'] = self._extract_base_id(img_id)
        
        return data_info
    
    def _extract_base_id(self, img_id):
        """Extract base ID without modality keyword.
        
        Examples:
            set00_V000_visible_I01216 → set00_V000_I01216
            set00_V000_lwir_I01216 → set00_V000_I01216
        
        Returns:
            str: Base ID that can be used to find paired modality.
        """
        # Remove modality keywords
        base_id = img_id.replace('_visible', '').replace('_lwir', '')
        base_id = base_id.replace('_infrared', '').replace('_thermal', '')
        return base_id

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Supports two modes:
        1. Single modality (default): Returns one modality based on sample ID
        2. Paired modality: Returns both visible + infrared images
        
        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data with modality information.
                  If return_modality_pair=True, includes both modalities.
        """
        # [DEBUG] Print mode for first few samples
        if idx < 3:
            print(f"[KAISTDataset.__getitem__] idx={idx}, return_modality_pair={self.return_modality_pair}")
        
        if not self.return_modality_pair:
            # Standard single-modality mode
            data = super().__getitem__(idx)
            
            # Add modality info only when pipeline generates data_samples
            if 'data_samples' in data:
                img_path = data['data_samples'].img_path
                if 'visible' in img_path.replace('\\', '/').lower():
                    modality = 'visible'
                elif any(x in img_path.replace('\\', '/').lower() for x in ['lwir', 'infrared', 'thermal']):
                    modality = 'infrared'
                else:
                    modality = 'unknown'
                
                data['data_samples'].metainfo['modality'] = modality
            
            return data
        
        else:
            # Paired modality mode
            return self._get_paired_data(idx)
    
    def _get_paired_data(self, idx):
        """Load paired visible + infrared images for the same scene.
        
        Process both modalities through the pipeline and return structured data.
        CRITICAL: Each modality must have its own DEEP COPY of instances to avoid
        shared reference issues during bbox transformations.
        
        Args:
            idx (int): Index of data.
            
        Returns:
            dict: Pipeline-processed data with both modalities embedded.
        """
        import copy
        
        # [DEBUG] Print for first few calls
        if idx < 3:
            print(f"[KAISTDataset._get_paired_data] Called for idx={idx}")
        
        # Get base sample info (visible)
        data_info = self.get_data_info(idx)
        base_id = data_info['base_id']
        
        # Find indices for both modalities
        visible_idx = None
        infrared_idx = None
        
        for i, info in enumerate(self.data_list):
            if info['base_id'] == base_id:
                if info['modality'] == 'visible':
                    visible_idx = i
                elif info['modality'] == 'infrared':
                    infrared_idx = i
        
        # If we can't find both, fall back to current index processing
        if visible_idx is None or infrared_idx is None:
            # Fallback: process current sample normally but mark as unpaired
            results = super().__getitem__(idx)
            if 'data_samples' in results:
                results['data_samples'].metainfo['modality'] = 'unpaired_fallback'
            return results
        
        # Process visible modality through pipeline
        visible_data_info = self.get_data_info(visible_idx)
        # CRITICAL: Deep copy to ensure visible has its own instances object
        visible_data_info = copy.deepcopy(visible_data_info)
        visible_results = self.pipeline(visible_data_info)
        
        # Process infrared modality through pipeline  
        infrared_data_info = self.get_data_info(infrared_idx)
        # CRITICAL: Deep copy to ensure infrared has its own instances object
        infrared_data_info = copy.deepcopy(infrared_data_info)
        infrared_results = self.pipeline(infrared_data_info)
        
        # Attach infrared image tensor to visible results' data_samples
        # This allows the detector to access both modalities
        if 'data_samples' in visible_results and 'inputs' in infrared_results:
            # [DEBUG] Print for first few calls
            if idx < 3:
                print(f"[KAISTDataset._get_paired_data] Attaching infrared_img for idx={idx}, shape={infrared_results['inputs'].shape if hasattr(infrared_results['inputs'], 'shape') else 'N/A'}")
            
            visible_results['data_samples'].infrared_img = infrared_results['inputs']
            visible_results['data_samples'].metainfo['modality'] = 'paired'
            visible_results['data_samples'].metainfo['base_id'] = base_id
            
            # [DEBUG] Verify attachment
            if idx < 3:
                has_it = hasattr(visible_results['data_samples'], 'infrared_img')
                print(f"[KAISTDataset._get_paired_data] Verification: has infrared_img = {has_it}")
                if has_it:
                    print(f"                                  infrared_img id = {id(visible_results['data_samples'].infrared_img)}")
        
        return visible_results
