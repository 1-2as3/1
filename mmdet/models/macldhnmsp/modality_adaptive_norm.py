# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class ModalityAdaptiveNorm(nn.Module):
    """Modality-Adaptive Normalization layer.
    
    This module uses separate batch normalization layers for different
    modalities (visible and infrared), allowing the network to learn
    modality-specific feature statistics.
    
    Args:
        channels (int): Number of input channels.
        momentum (float): Momentum for running mean/var. Defaults to 0.1.
        eps (float): Small value to avoid division by zero. Defaults to 1e-5.
    """
    
    def __init__(self, channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.channels = channels
        
        # 为可见光模态创建 BN 层
        self.bn_vis = nn.BatchNorm2d(channels, momentum=momentum, eps=eps)
        
        # 为红外模态创建 BN 层
        self.bn_ir = nn.BatchNorm2d(channels, momentum=momentum, eps=eps)
        
        # 默认 BN 层（用于未知模态）
        self.bn_default = nn.BatchNorm2d(channels, momentum=momentum, eps=eps)
    
    def forward(self, x, modality=None):
        """Forward pass with modality-specific normalization.
        
        Args:
            x (Tensor): Input feature maps, shape (N, C, H, W).
            modality (str or Tensor): Modality indicator.
                - If str: 'visible', 'infrared', or other
                - If Tensor: batch of modality labels
                - If None: use default BN
                
        Returns:
            Tensor: Normalized features, same shape as input.
        """
        if modality is None:
            return self.bn_default(x)
        
        # 如果 modality 是字符串（整个 batch 同一模态）
        if isinstance(modality, str):
            if 'visible' in modality.lower() or 'vis' in modality.lower():
                return self.bn_vis(x)
            elif 'infrared' in modality.lower() or 'ir' in modality.lower() or \
                 'lwir' in modality.lower() or 'thermal' in modality.lower():
                return self.bn_ir(x)
            else:
                return self.bn_default(x)
        
        # 如果 modality 是 Tensor（batch 中可能有不同模态）
        elif torch.is_tensor(modality):
            # 假设 modality 是形状为 (N,) 的整数 tensor
            # 0: visible, 1: infrared
            output = torch.zeros_like(x)
            for i in range(x.size(0)):
                if modality[i] == 0:
                    output[i] = self.bn_vis(x[i:i+1])
                elif modality[i] == 1:
                    output[i] = self.bn_ir(x[i:i+1])
                else:
                    output[i] = self.bn_default(x[i:i+1])
            return output
        
        # 如果是 list（batch 中的模态字符串列表）
        elif isinstance(modality, (list, tuple)):
            output = torch.zeros_like(x)
            for i, mod in enumerate(modality):
                if 'visible' in str(mod).lower() or 'vis' in str(mod).lower():
                    output[i] = self.bn_vis(x[i:i+1])
                elif 'infrared' in str(mod).lower() or 'ir' in str(mod).lower():
                    output[i] = self.bn_ir(x[i:i+1])
                else:
                    output[i] = self.bn_default(x[i:i+1])
            return output
        
        else:
            return self.bn_default(x)


@MODELS.register_module()
class ModalityAdaptiveResNet(nn.Module):
    """ResNet backbone with Modality-Adaptive Normalization.
    
    This is a wrapper around standard ResNet that adds modality-adaptive
    normalization to the first two layers.
    
    Args:
        backbone_cfg (dict): Config for the base ResNet backbone.
        use_modality_bn (bool): Whether to use modality-adaptive BN.
            Defaults to True.
        modality_bn_layers (list): Which layers to apply modality BN.
            Defaults to [0, 1] (first two layers).
    """
    
    def __init__(self, 
                 backbone_cfg,
                 use_modality_bn=True,
                 modality_bn_layers=None):
        super().__init__()
        
        if modality_bn_layers is None:
            modality_bn_layers = [0, 1]
        
        self.use_modality_bn = use_modality_bn
        self.modality_bn_layers = modality_bn_layers
        
        # 构建基础 backbone
        self.backbone = MODELS.build(backbone_cfg)
        
        # 为指定层添加 modality-adaptive BN
        if use_modality_bn:
            self.modality_bns = nn.ModuleDict()
            
            # 假设 ResNet 的层结构：layer0 (stem), layer1, layer2, layer3, layer4
            layer_channels = {
                0: 64,   # layer0/stem 输出通道
                1: 256,  # layer1 输出通道 (ResNet50)
                2: 512,  # layer2 输出通道
                3: 1024, # layer3 输出通道
                4: 2048  # layer4 输出通道
            }
            
            for layer_idx in modality_bn_layers:
                if layer_idx in layer_channels:
                    self.modality_bns[f'layer{layer_idx}'] = \
                        ModalityAdaptiveNorm(layer_channels[layer_idx])
    
    def forward(self, x, modality=None):
        """Forward with modality information.
        
        Args:
            x (Tensor): Input images.
            modality (str, Tensor, or list): Modality information.
            
        Returns:
            tuple: Multi-scale features from backbone.
        """
        # 如果没有提供模态信息，使用标准 forward
        if not self.use_modality_bn or modality is None:
            return self.backbone(x)
        
        # 需要修改 backbone 的 forward 来支持中间插入 modality BN
        # 这里提供一个示例实现
        # 实际使用时需要根据具体的 backbone 结构调整
        
        # 简化示例：假设可以访问 backbone 的各层
        outs = []
        
        # Layer 0 (stem)
        x = self.backbone.conv1(x)
        if 0 in self.modality_bn_layers:
            x = self.modality_bns['layer0'](x, modality)
        else:
            x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Layer 1
        x = self.backbone.layer1(x)
        if 1 in self.modality_bn_layers:
            x = self.modality_bns['layer1'](x, modality)
        outs.append(x)
        
        # Layer 2-4 (标准处理)
        for i, layer in enumerate([self.backbone.layer2, 
                                   self.backbone.layer3, 
                                   self.backbone.layer4], start=2):
            x = layer(x)
            if i in self.modality_bn_layers:
                x = self.modality_bns[f'layer{i}'](x, modality)
            outs.append(x)
        
        return tuple(outs)
