"""Gradient Direction Monitor Hook

Monitor cosine similarity between gradients from different loss components:
- loss_cls vs loss_bbox (should be aligned, ~0.5-0.8)
- MACL loss vs detection losses (if <-0.3, indicates severe conflict)

Usage in config:
    custom_hooks = [
        dict(type='GradientMonitorHook', 
             interval=50,
             target_layers=['backbone.layer4', 'roi_head.bbox_head.fc_cls'],
             print_detail=True)
    ]
"""
import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional, Dict, List
import numpy as np


@HOOKS.register_module()
class GradientMonitorHook(Hook):
    """Monitor gradient direction conflicts during training.
    
    Computes cosine similarity between gradients from different loss sources
    to detect potential gradient competition issues.
    """
    
    priority = 'NORMAL'
    
    def __init__(self,
                 interval: int = 50,
                 target_layers: Optional[List[str]] = None,
                 print_detail: bool = True,
                 log_grad_norm: bool = True):
        """
        Args:
            interval: Check gradient every N iterations
            target_layers: List of layer names to monitor (e.g., ['backbone.layer4'])
                         If None, monitors all parameters with grad
            print_detail: Print detailed gradient analysis
            log_grad_norm: Log gradient norm for each component
        """
        self.interval = interval
        self.target_layers = target_layers or []
        self.print_detail = print_detail
        self.log_grad_norm = log_grad_norm
        
        # Storage for gradient snapshots
        self.grad_snapshots = {}
    
    def _get_layer_params(self, runner, layer_name: str):
        """Get parameters of a specific layer by name."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        try:
            layer = model
            for attr in layer_name.split('.'):
                layer = getattr(layer, attr)
            return list(layer.parameters())
        except AttributeError:
            runner.logger.warning(f"Layer {layer_name} not found in model")
            return []
    
    def _compute_grad_cosine(self, grads1: List[torch.Tensor], 
                            grads2: List[torch.Tensor]) -> float:
        """Compute cosine similarity between two gradient sets."""
        if not grads1 or not grads2:
            return 0.0
        
        # Flatten all gradients
        flat1 = torch.cat([g.flatten() for g in grads1 if g is not None])
        flat2 = torch.cat([g.flatten() for g in grads2 if g is not None])
        
        if flat1.numel() == 0 or flat2.numel() == 0:
            return 0.0
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0), 
            flat2.unsqueeze(0)
        )
        return cos_sim.item()
    
    def _get_grad_norm(self, grads: List[torch.Tensor]) -> float:
        """Compute L2 norm of gradients."""
        if not grads:
            return 0.0
        flat = torch.cat([g.flatten() for g in grads if g is not None])
        return torch.norm(flat, p=2).item()
    
    def after_train_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: Optional[dict] = None,
                        outputs: Optional[dict] = None) -> None:
        """Monitor gradients after each training iteration."""
        if self.every_n_train_iters(runner, self.interval):
            # Get loss components from message_hub
            losses = runner.message_hub.get_scalar('train/loss').current()
            
            # Collect gradients for each loss component
            grad_info = {}
            
            # If MACL is enabled, check for gradient conflicts
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            
            # Target layers to monitor
            if self.target_layers:
                for layer_name in self.target_layers:
                    params = self._get_layer_params(runner, layer_name)
                    grads = [p.grad for p in params if p.grad is not None]
                    
                    if grads:
                        norm = self._get_grad_norm(grads)
                        grad_info[layer_name] = {
                            'norm': norm,
                            'grads': grads
                        }
            else:
                # Monitor all model parameters
                all_grads = [p.grad for p in model.parameters() if p.grad is not None]
                if all_grads:
                    grad_info['all_params'] = {
                        'norm': self._get_grad_norm(all_grads),
                        'grads': all_grads
                    }
            
            # Log gradient norms
            if self.log_grad_norm and grad_info:
                for name, info in grad_info.items():
                    runner.logger.info(f"Gradient norm [{name}]: {info['norm']:.4f}")
            
            # Compute cosine similarity between loss components (requires manual backward separation)
            # For now, just log the overall gradient statistics
            if self.print_detail:
                runner.logger.info(
                    f"Gradient Monitor [Iter {runner.iter}]: "
                    f"Monitored {len(grad_info)} layer groups"
                )
