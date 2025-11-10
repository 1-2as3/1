from mmdet.registry import DATASETS
from mmdet.datasets.voc import VOCDataset
from mmdet.structures import DetDataSample
import os.path as osp


@DATASETS.register_module()
class LLVIPDataset(VOCDataset):
    METAINFO = {'classes': ('person',), 'dataset_type': 'LLVIP'}
    
    def __init__(self, *args, return_modality_pair=False, **kwargs):
        """Initialize LLVIP Dataset.
        
        Args:
            return_modality_pair (bool): If True, return paired (visible, infrared) inputs.
                Used for cross-modal representation learning. Defaults to False.
        """
        self.return_modality_pair = return_modality_pair
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # If not in paired mode, use original logic with modality detection
        if not self.return_modality_pair:
            data_sample = data.get('data_samples', None)
            if data_sample is None:
                return data

            # Attempt to read image path from common fields
            img_path = None
            if hasattr(data_sample, 'img_path'):
                img_path = data_sample.img_path
            else:
                metainfo = getattr(data_sample, 'metainfo', {})
                img_path = metainfo.get('img_path', None)

            modality = 'unknown'
            if isinstance(img_path, str):
                low = img_path.replace('\\', '/').lower()
                if 'visible' in low or '/vis' in low or '/rgb' in low:
                    modality = 'visible'
                elif 'infrared' in low or 'ir' in low or 'thermal' in low:
                    modality = 'infrared'

            # write modality into DetDataSample.metainfo
            try:
                data_sample.set_metainfo({**getattr(data_sample, 'metainfo', {}), 'modality': modality})
            except Exception:
                try:
                    data_sample.metainfo['modality'] = modality
                except Exception:
                    pass

            return data
        
        # Paired modality mode - load both but return in standard format
        # Get the data sample and extract image path info
        data_sample = data.get('data_samples', None)
        if data_sample is None:
            return data
        
        # Get the original image path from metainfo
        metainfo = getattr(data_sample, 'metainfo', {})
        img_path = metainfo.get('img_path', None)
        
        if img_path is None:
            # Fallback: return original data
            return data
        
        # Construct paired paths
        # Assume structure: .../visible/train/xxx.jpg and .../infrared/train/xxx.jpg
        img_path = img_path.replace('\\', '/')
        
        if 'visible' in img_path:
            vis_path = img_path
            ir_path = img_path.replace('visible', 'infrared')
        elif 'infrared' in img_path:
            ir_path = img_path
            vis_path = img_path.replace('infrared', 'visible')
        else:
            # Cannot determine paths, return original
            return data
        
        # Check if both files exist
        if not (osp.exists(vis_path) and osp.exists(ir_path)):
            # Fallback to single modality
            return data
        
        # Load infrared image using the pipeline
        # Create a temporary data dict for IR processing
        ir_data_info = self.get_data_info(idx)
        ir_data_info['img_path'] = ir_path
        
        # Process through pipeline to get IR tensor
        ir_processed = self.pipeline(ir_data_info)
        ir_inputs = ir_processed['inputs']
        
        # Store IR tensor in data_sample for later use in model
        # Use a custom field to avoid conflicts
        try:
            # Store as a tensor that will be moved to device along with data_sample
            import torch
            if not isinstance(ir_inputs, torch.Tensor):
                ir_inputs = torch.from_numpy(ir_inputs)
            # Attach to data_sample - will be accessible in model forward
            data_sample.set_field(ir_inputs, 'infrared_img', dtype=torch.Tensor)
        except Exception as e:
            print(f"[WARN] Failed to attach infrared image: {e}")
        
        # Update data sample metainfo
        try:
            data_sample.set_metainfo({
                **metainfo,
                'modality_pair': True,
                'vis_img_path': vis_path,
                'ir_img_path': ir_path
            })
        except Exception:
            pass
        
        # Return in standard format: visible as inputs, infrared stored in data_sample
        return dict(
            inputs=data['inputs'],  # visible image (standard format)
            data_samples=data_sample
        )
