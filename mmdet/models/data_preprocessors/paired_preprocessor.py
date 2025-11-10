# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType
from .data_preprocessor import DetDataPreprocessor


@MODELS.register_module()
class PairedDetDataPreprocessor(DetDataPreprocessor):
    """Data preprocessor for paired modality (visible + infrared) detection.
    
    Extends DetDataPreprocessor to handle both visible and infrared images:
    - Extracts infrared images from data_samples.infrared_img
    - Converts both modalities to float32 and normalizes with same mean/std
    - Pads and stacks both modalities independently
    - Returns dict with 'visible' and 'infrared' keys in batch_inputs
    - Updates data_samples meta info (pad_shape, img_shape, etc.)
    
    Args:
        mean (Sequence[float], optional): Mean values for normalization.
        std (Sequence[float], optional): Std values for normalization.
        pad_size_divisor (int): Pad spatial size to multiple of this value.
        pad_value (float): Padding value. Defaults to 0.
        bgr_to_rgb (bool): Whether to convert BGR to RGB.
        rgb_to_bgr (bool): Whether to convert RGB to BGR.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with base class to set up _channel_conversion and other helpers."""
        super().__init__(*args, **kwargs)
        # Ensure _channel_conversion is callable (base class should set this, but double-check)
        if not callable(getattr(self, '_channel_conversion', None)):
            # Fallback: set identity function if base class didn't initialize it properly
            self._channel_conversion = lambda x: x

    def forward(self, data: dict, training: bool = False) -> dict:
        """Unified preprocess for visible and infrared modalities.
        
        Args:
            data (dict): Data from dataloader. Contains:
                - 'inputs': List[Tensor], visible images (uint8 or float)
                - 'data_samples': List[DetDataSample], with .infrared_img
            training (bool): Whether in training mode.
            
        Returns:
            dict: Preprocessed data:
                - 'inputs': dict with 'visible' and 'infrared' batched tensors
                - 'data_samples': List[DetDataSample] with updated meta info
        """
        # Extract inputs and data_samples
        visible_inputs = data['inputs']  # List[Tensor]
        data_samples = data.get('data_samples', None)
        
        # Check if infrared images are present
        has_infrared = (
            data_samples is not None 
            and len(data_samples) > 0 
            and hasattr(data_samples[0], 'infrared_img')
            and data_samples[0].infrared_img is not None
        )
        
        if has_infrared:
            # Extract infrared images from data_samples
            infrared_inputs = [sample.infrared_img for sample in data_samples]
            
            # Process visible modality
            visible_data = self._process_modality(
                visible_inputs, data_samples, training, modality='visible'
            )
            
            # Process infrared modality (reuse same data_samples for meta updates)
            infrared_data = self._process_modality(
                infrared_inputs, data_samples, training, modality='infrared'
            )
            
            # Combine into paired batch_inputs
            batch_inputs = {
                'visible': visible_data['inputs'],
                'infrared': infrared_data['inputs']
            }
            
            return {
                'inputs': batch_inputs,
                'data_samples': visible_data['data_samples']  # meta updated from visible
            }
        else:
            # No infrared: fallback to standard single-modality preprocessing
            processed = self._process_modality(
                visible_inputs, data_samples, training, modality='visible'
            )
            return {
                'inputs': processed['inputs'],
                'data_samples': processed['data_samples']
            }
    
    def _process_modality(
        self, 
        inputs: List[Tensor], 
        data_samples: Optional[List[DetDataSample]],
        training: bool,
        modality: str
    ) -> dict:
        """Process a single modality (visible or infrared).
        
        Args:
            inputs (List[Tensor]): List of images for this modality.
            data_samples (List[DetDataSample], optional): Data samples.
            training (bool): Training mode flag.
            modality (str): 'visible' or 'infrared'.
            
        Returns:
            dict: Processed inputs and data_samples.
        """
        # Move to device and convert to float
        inputs = [img.to(self.device) for img in inputs]
        inputs = [self._channel_conversion(img) for img in inputs]
        inputs = [img.float() for img in inputs]
        
        # Normalize if mean/std configured
        if self._enable_normalize:
            inputs = [self._normalize(img) for img in inputs]
        
        # Pad to same size
        if self.pad_size_divisor > 1:
            inputs = [self._pad_image(img) for img in inputs]
        
        # Stack into batch tensor
        batch_inputs = torch.stack(inputs, dim=0)
        
        # Update meta info (only for visible modality to avoid overwriting)
        if data_samples is not None and modality == 'visible':
            batch_input_shape = tuple(batch_inputs.shape[-2:])
            for sample, img in zip(data_samples, inputs):
                sample.set_metainfo({
                    'img_shape': tuple(img.shape[-2:]),
                    'pad_shape': batch_input_shape,
                    'batch_input_shape': batch_input_shape
                })
                # Move gt_instances to device if present
                if hasattr(sample, 'gt_instances'):
                    sample.gt_instances = sample.gt_instances.to(self.device)
        
        return {
            'inputs': batch_inputs,
            'data_samples': data_samples
        }
    
    def _normalize(self, img: Tensor) -> Tensor:
        """Normalize image with configured mean and std."""
        mean = self.mean.view(-1, 1, 1)
        std = self.std.view(-1, 1, 1)
        return (img - mean) / std
    
    def _pad_image(self, img: Tensor) -> Tensor:
        """Pad image to be divisible by pad_size_divisor."""
        h, w = img.shape[-2:]
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=self.pad_value)
        
        return img
