# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation.
    
    This layer acts as an identity function during forward pass,
    but reverses the gradient during backward pass, which enables
    adversarial domain adaptation.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass - identity function.
        
        Args:
            ctx: Context object to save information for backward pass.
            x (Tensor): Input tensor.
            lambda_ (float): Gradient reversal strength.
            
        Returns:
            Tensor: Output tensor (same as input).
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - reverse and scale gradient.
        
        Args:
            ctx: Context object with saved information.
            grad_output (Tensor): Gradient from upstream.
            
        Returns:
            tuple: (reversed gradient, None for lambda_)
        """
        return -ctx.lambda_ * grad_output, None


@MODELS.register_module()
class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation.
    
    This module tries to predict which domain (dataset) a feature comes from.
    When combined with GradientReversalLayer, it encourages domain-invariant
    feature learning.
    
    Args:
        in_dim (int): Input feature dimension. Defaults to 256.
        num_domains (int): Number of domains/datasets. Defaults to 3.
        hidden_dim (int): Hidden layer dimension. Defaults to 128.
    """
    
    def __init__(self, in_dim=256, num_domains=3, hidden_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.num_domains = num_domains
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 添加 dropout 防止过拟合
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, x, lambda_=1.0):
        """Forward pass with gradient reversal.
        
        Args:
            x (Tensor): Input features, shape (N, in_dim).
            lambda_ (float): Gradient reversal strength. Defaults to 1.0.
                Higher values mean stronger domain confusion.
                
        Returns:
            Tensor: Domain logits, shape (N, num_domains).
        """
        # 如果输入是 4D 特征图，进行全局平均池化
        if x.dim() == 4:
            x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        
        # 应用梯度反转
        x = GradientReversalLayer.apply(x, lambda_)
        
        # 通过分类器
        return self.fc(x)
