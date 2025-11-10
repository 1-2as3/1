"""
T-SNE 可视化 Hook：展示 MACL 跨模态 embedding 对齐效果
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@HOOKS.register_module()
class TSNEVisualHook(Hook):
    """跨模态 embedding 可视化 Hook (t-SNE)
    
    用于可视化 MACL 模块学习到的可见光和红外 embedding 的分布对齐情况。
    
    Args:
        out_dir (str): 输出目录。默认 'work_dirs/tsne_vis'
        interval (int): 可视化间隔（epoch）。默认 1
        num_samples (int): 每个模态采样数量。默认 100
        vis_classes (tuple): 模态类别名称。默认 ('visible', 'infrared')
        random_seed (int): 随机种子。默认 42
        perplexity (int): t-SNE perplexity 参数。默认 30
        enable_3d (bool): 是否生成 3D 可视化。默认 False
    """
    
    priority = 'NORMAL'
    
    def __init__(self,
                 out_dir: str = 'work_dirs/tsne_vis',
                 interval: int = 1,
                 num_samples: int = 100,
                 vis_classes: Tuple[str, str] = ('visible', 'infrared'),
                 random_seed: int = 42,
                 perplexity: int = 30,
                 enable_3d: bool = False) -> None:
        if not HAS_SKLEARN:
            print('[WARN] scikit-learn not installed, TSNEVisualHook disabled. '
                  'Install with: pip install scikit-learn')
            self.enabled = False
            return
        
        self.out_dir = out_dir
        self.interval = interval
        self.num_samples = num_samples
        self.vis_classes = vis_classes
        self.random_seed = random_seed
        self.perplexity = perplexity
        self.enable_3d = enable_3d
        self.enabled = True
        
        os.makedirs(out_dir, exist_ok=True)
        np.random.seed(random_seed)
        
    def after_train_epoch(self, runner) -> None:
        """每个 epoch 结束后生成 t-SNE 可视化"""
        if not self.enabled:
            return
            
        epoch = runner.epoch
        
        # 检查间隔
        if epoch % self.interval != 0:
            return
        
        # 获取 MACL embeddings
        embeddings = self._extract_embeddings(runner)
        if embeddings is None:
            return
        
        vis_embeds, ir_embeds = embeddings
        
        # 采样
        n_vis = min(self.num_samples, len(vis_embeds))
        n_ir = min(self.num_samples, len(ir_embeds))
        
        if n_vis == 0 or n_ir == 0:
            runner.logger.warning('[TSNEVisualHook] No embeddings to visualize')
            return
        
        # 随机采样
        vis_idx = np.random.choice(len(vis_embeds), n_vis, replace=False)
        ir_idx = np.random.choice(len(ir_embeds), n_ir, replace=False)
        
        vis_sample = vis_embeds[vis_idx]
        ir_sample = ir_embeds[ir_idx]
        
        # 合并数据
        X = np.concatenate([vis_sample, ir_sample], axis=0)
        y = np.array([0] * n_vis + [1] * n_ir)
        
        # 生成 2D t-SNE
        try:
            self._plot_2d_tsne(X, y, epoch, runner)
            runner.logger.info(f'[TSNEVisualHook] Generated t-SNE visualization for epoch {epoch}')
        except Exception as e:
            runner.logger.warning(f'[TSNEVisualHook] Failed to generate t-SNE: {e}')
    
    def _extract_embeddings(self, runner) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """提取 MACL embeddings"""
        model = runner.model
        
        # 兼容 DP/DDP
        if hasattr(model, 'module'):
            model = model.module
        
        # 检查是否有 roi_head
        if not hasattr(model, 'roi_head'):
            return None
        
        roi_head = model.roi_head
        
        # 检查是否有 macl_head
        if not hasattr(roi_head, 'macl_head'):
            return None
        
        macl_head = roi_head.macl_head
        
        # 优先使用聚合缓冲（一个 epoch 多批次），否则退回 latest_embeddings
        vis_embeds = None
        ir_embeds = None
        # 优先从缓冲区弹出
        if hasattr(macl_head, 'pop_buffer_embeddings'):
            popped = macl_head.pop_buffer_embeddings()
            if isinstance(popped, dict):
                vis_embeds = popped.get('vis', None)
                ir_embeds = popped.get('ir', None)
        # 如果缓冲不足，使用 latest_embeddings（单批次）
        if (vis_embeds is None or ir_embeds is None) and hasattr(macl_head, 'latest_embeddings'):
            embeddings = macl_head.latest_embeddings
            if isinstance(embeddings, dict):
                vis_embeds = embeddings.get('vis', None)
                ir_embeds = embeddings.get('ir', None)
        if vis_embeds is None or ir_embeds is None:
            return None
        
        # 转换为 numpy
        if torch.is_tensor(vis_embeds):
            vis_embeds = vis_embeds.detach().cpu().numpy()
        if torch.is_tensor(ir_embeds):
            ir_embeds = ir_embeds.detach().cpu().numpy()
        
        return vis_embeds, ir_embeds
    
    def _plot_2d_tsne(self, X: np.ndarray, y: np.ndarray, epoch: int, runner) -> None:
        """绘制 2D t-SNE 图"""
        # 计算 t-SNE
        perplexity = min(self.perplexity, len(X) - 1)
        tsne = TSNE(n_components=2, random_state=self.random_seed, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        # 绘制散点图
        plt.figure(figsize=(8, 8))
        
        # 可见光 - 蓝色
        vis_mask = (y == 0)
        plt.scatter(X_tsne[vis_mask, 0], X_tsne[vis_mask, 1], 
                   c='royalblue', label=self.vis_classes[0].capitalize(), 
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # 红外 - 红色
        ir_mask = (y == 1)
        plt.scatter(X_tsne[ir_mask, 0], X_tsne[ir_mask, 1], 
                   c='orangered', label=self.vis_classes[1].capitalize(), 
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        plt.legend(fontsize=12, frameon=True, shadow=True)
        plt.title(f't-SNE of MACL Embeddings (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 保存
        out_path = os.path.join(self.out_dir, f'tsne_epoch{epoch}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        runner.logger.info(f'[OK] t-SNE visualization saved: {out_path}')
        
        # 计算并记录对齐指标
        self._compute_alignment_metrics(X_tsne, y, epoch, runner)
    
    def _compute_alignment_metrics(self, X_tsne: np.ndarray, y: np.ndarray, 
                                   epoch: int, runner) -> None:
        """计算并记录对齐指标"""
        try:
            from scipy.spatial.distance import cdist
            
            # 计算类内和类间距离
            vis_points = X_tsne[y == 0]
            ir_points = X_tsne[y == 1]
            # 样本过少则跳过，避免 0 或噪声
            if len(vis_points) < 5 or len(ir_points) < 5:
                runner.logger.warning('[TSNEVisualHook] Too few samples for reliable alignment metrics, skipped')
                return
            
            # 类内平均距离（去掉对角项）
            d_vis = cdist(vis_points, vis_points)
            np.fill_diagonal(d_vis, np.nan)
            vis_intra = np.nanmean(d_vis)
            d_ir = cdist(ir_points, ir_points)
            np.fill_diagonal(d_ir, np.nan)
            ir_intra = np.nanmean(d_ir)
            
            # 类间平均距离
            inter = cdist(vis_points, ir_points).mean()
            
            # 对齐分数（类间距离越小、类内距离越大，对齐越好）
            denom = (vis_intra + ir_intra)
            if not np.isfinite(denom) or denom <= 0:
                runner.logger.warning('[TSNEVisualHook] Intra distance invalid (<=0). Skip alignment score')
                return
            alignment_score = inter / denom
            
            runner.logger.info(
                f'[TSNEVisualHook] Alignment metrics - '
                f'Inter-modal dist: {inter:.4f}, '
                f'Intra-modal dist (vis/ir): {vis_intra:.4f}/{ir_intra:.4f}, '
                f'Alignment score: {alignment_score:.4f}'
            )
            
        except ImportError:
            pass  # scipy not available
        except Exception as e:
            runner.logger.warning(f'[TSNEVisualHook] Failed to compute metrics: {e}')
