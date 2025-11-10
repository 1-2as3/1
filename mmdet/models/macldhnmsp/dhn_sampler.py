import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class DHNSampler(nn.Module):
    """Dynamic Hard Negative Sampler with momentum queue.
    
    Args:
        dim (int): Feature dimension.
        queue_size (int): Size of the momentum queue.
        momentum (float): Momentum coefficient.
        top_k (int): Number of hard negatives to sample.
    """
    
    def __init__(self,
                 dim=128,
                 queue_size=8192,
                 momentum=0.99,
                 top_k=256):
        super().__init__()
        self.dim = dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.top_k = min(top_k, queue_size)
        
        # 初始化动量队列
        self.register_buffer('queue', torch.randn(queue_size, dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def update_queue(self, features):
        """Update the momentum queue with new features.
        
        Args:
            features (Tensor): New features to enqueue, shape (N, dim).
        """
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的特征
        if ptr + batch_size > self.queue_size:
            # 处理环形队列边界
            first_part = self.queue_size - ptr
            self.queue[ptr:] = features[:first_part]
            self.queue[:batch_size - first_part] = features[first_part:]
            ptr = batch_size - first_part
        else:
            self.queue[ptr:ptr + batch_size] = features
            ptr = ptr + batch_size if ptr + batch_size < self.queue_size else 0
            
        self.queue_ptr[0] = ptr
        
        # 动量更新
        self.queue = self.momentum * self.queue + (1 - self.momentum) * self.queue.clone()
        self.queue = F.normalize(self.queue, dim=1)
        
    def sample_hard_negatives(self, query_features):
        """Sample hard negative examples based on similarity.
        
        Args:
            query_features (Tensor): Query features to find hard negatives.
            
        Returns:
            Tensor: Selected hard negative samples.
        """
        with torch.no_grad():
            # 计算查询特征与队列中所有特征的相似度
            sim = torch.matmul(query_features, self.queue.T)
            
            # 选择 top-k 困难负样本
            _, indices = torch.topk(sim, k=self.top_k, dim=1)
            
            # 获取选中的负样本特征
            hard_negatives = self.queue[indices]
            
        return hard_negatives
    
    def compute_loss(self, *args, **kwargs):
        """Placeholder loss computation method for compatibility.
        
        This method is called from StandardRoIHead when use_dhn=True.
        DHN loss is typically computed within MACLHead's forward pass.
        
        Returns:
            dict: Empty dict as placeholder.
        """
        # DHN 的损失通常在 MACLHead 中计算，这里提供占位接口
        return {}
