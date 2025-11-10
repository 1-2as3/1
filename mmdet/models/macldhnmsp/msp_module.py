import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class MSPReweight(nn.Module):
    """Multi-Scale Pattern Reweighting Module.

    Args:
        channels (int, optional): Preferred channels parameter.
        in_channels (int, optional): Backwards-compatible alias for input channels.
        out_channels (int, optional): Backwards-compatible alias for output channels.
        alpha (float): Reweighting factor.
    """
    def __init__(self, channels: int = None, in_channels: int = None, 
                 out_channels: int = None, alpha: float = 0.5):
        super().__init__()
        # Accept any of channels / out_channels / in_channels for compatibility
        self.channels = int(channels or out_channels or in_channels or 256)
        
        # 初始化可学习的 alpha 参数
        self.alpha = nn.Parameter(torch.ones(1) * alpha)

        # 1x1 卷积层用于生成注意力图
        self.conv_avg = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.conv_max = nn.Conv2d(self.channels, 1, kernel_size=1)
        
        # 用于通道不匹配时的投影卷积层字典，延迟初始化
        self.channel_projectors = {}
        
    def forward(self, inputs):
        """Forward function.
        
        Args:
            inputs (tuple[Tensor]): List of multi-level feature maps.
            
        Returns:
            tuple[Tensor]: Enhanced feature maps.
        """
        assert isinstance(inputs, (list, tuple))
        outs = []
        
        # 初始化层级 alpha 参数（如果尚未初始化）
        if self.alpha is None:
            self.initialize_alpha(len(inputs))
        
        for idx, feat in enumerate(inputs):
            # 处理通道不匹配情况
            if feat.shape[1] != self.channels:
                # 创建或获取投影器
                projector_key = f"{feat.shape[1]}->{self.channels}"
                if projector_key not in self.channel_projectors:
                    # 创建新的投影器并缓存
                    self.channel_projectors[projector_key] = nn.Conv2d(
                        feat.shape[1], self.channels, kernel_size=1).to(feat.device)
                # 使用缓存的投影器
                feat_proj = self.channel_projectors[projector_key](feat)
            else:
                feat_proj = feat

            # 使用预先定义的卷积层计算注意力图
            avg_out = self.conv_avg(feat_proj)
            max_out = self.conv_max(feat_proj)

            # generate attention and apply sigmoid
            attention = torch.sigmoid(avg_out + max_out)

            # 应用 sigmoid 来约束 alpha 在 [0,1] 范围内
            alpha = torch.sigmoid(self.alpha)

            # 应用 sigmoid 来约束 alpha 在 [0,1] 范围内
            alpha = torch.sigmoid(self.alpha)

            # feature enhancement: F'_l = F_l * (1 + α * attention)
            if feat_proj is not feat:
                # attention has shape (N,1,H,W); map it back to original channels
                enhanced = feat * (1 + alpha * F.interpolate(attention, size=feat.shape[2:], mode='nearest'))
            else:
                enhanced = feat * (1 + alpha * attention)
                
            # 记录当前使用的 alpha 值以便后续监控
            if hasattr(self, '_log_alpha') and self.training:
                self._log_alpha = alpha.item()
                
            outs.append(enhanced)
            
        return tuple(outs)
    
    def get_alpha(self):
        """Get the current alpha value (after sigmoid).
        
        Returns:
            float: The current alpha value between 0 and 1.
        """
        with torch.no_grad():
            return torch.sigmoid(self.alpha).item()
            
    def get_loss(self):
        """Calculate regularization loss for alpha parameter.
        
        Returns:
            dict: A dictionary containing the regularization loss.
        """
        if hasattr(self, 'alpha'):
            loss_reg = 1e-4 * (self.alpha ** 2).sum()
            return {'loss_alpha_reg': loss_reg}
        return {}
    
    def compute_loss(self, *args, **kwargs):
        """Placeholder loss computation method for compatibility.
        
        This method is called from StandardRoIHead when use_msp=True.
        Delegates to get_loss() for actual alpha regularization.
        
        Returns:
            dict: Regularization loss for alpha parameter.
        """
        return self.get_loss()
