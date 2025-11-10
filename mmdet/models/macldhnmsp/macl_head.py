import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class MACLHead(nn.Module):
    """Modal-Aware Contrastive Learning Head.

    Args:
        in_dim (int): Input feature dimension.
        proj_dim (int): Projection dimension for contrastive learning.
        tau (float): Temperature parameter in InfoNCE loss (default: 0.07).
        temperature (float, optional): Alias for tau parameter. If provided, overrides tau.
        use_bn (bool): Whether to use batch normalization.
        use_dhn (bool): Whether to use DHN sampler for hard negatives.
        dhn_cfg (dict, optional): Configuration dict for DHNSampler. If None, uses default params.
    """

    def __init__(
        self,
        in_dim: int = 256,
        proj_dim: int = 128,
        tau: float = 0.07,
        temperature: float | None = None,
        use_bn: bool = True,
        use_dhn: bool = True,
        dhn_cfg: dict | None = None,
        # 新增：可配置归一化与投影深度/Dropout
        norm_type: str | None = None,  # 'BN' | 'GN' | None；优先于 use_bn
        norm_groups: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.0,
        # 域对齐损失（MMD）
        enable_domain_loss: bool = False,
        domain_weight: float = 0.1,
        mmd_kernels: tuple[float, ...] = (1.0, 2.0, 4.0),
    ):
        super().__init__()

        # 归一化层工厂（保持向后兼容 use_bn）
        def make_norm(c: int):
            nt = (norm_type or ("BN" if use_bn else None))
            if nt is None:
                return None
            nt = nt.upper()
            if nt == 'BN':
                return nn.BatchNorm1d(c)
            if nt == 'GN':
                # groups 不能超过通道数
                g = min(norm_groups, c)
                # groups 至少为1，且能整除通道数，不整除时回退为1
                if c % g != 0:
                    g = 1
                return nn.GroupNorm(g, c)
            # 未知类型，禁用归一化
            return None

        h1, h2 = hidden_dims
        layers: list[nn.Module] = []
        # 第一层
        layers += [nn.Linear(in_dim, h1)]
        n1 = make_norm(h1)
        if n1 is not None:
            layers += [n1]
        layers += [nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers += [nn.Dropout(p=float(dropout))]
        # 第二层
        layers += [nn.Linear(h1, h2)]
        n2 = make_norm(h2)
        if n2 is not None:
            layers += [n2]
        layers += [nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers += [nn.Dropout(p=float(dropout))]
        # 输出层
        layers += [nn.Linear(h2, proj_dim)]

        # 更深的投影头网络，增强特征表达能力和稳定性
        self.proj = nn.Sequential(*layers)
        # 使用 nn.Parameter 让 tau 可学习，初始值从 temperature 或 tau 获取
        # 支持 temperature 别名以兼容不同配置风格
        init_tau = temperature if temperature is not None else tau
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.use_dhn = use_dhn
        self.iter_count = 0  # 用于定期打印调试信息
        # 域对齐配置
        self.enable_domain_loss = enable_domain_loss
        self.domain_weight = float(domain_weight)
        self.mmd_kernels = tuple(float(s) for s in mmd_kernels)
        # 轻量级缓存，用于 TSNEVisualHook 聚合一个 epoch 内的多批次 embedding
        # Py3.11 某些环境下对行内类型标注解析有差异，使用注释形式
        self._embed_buf_vis = None  # type: torch.Tensor | None
        self._embed_buf_ir = None   # type: torch.Tensor | None
        self._embed_buf_max = 512   # 每个模态最多保留的样本数
        if self.use_dhn:
            try:
                from .dhn_sampler import DHNSampler
                # 如果提供了 dhn_cfg，用它实例化；否则使用默认参数
                # 支持旧式参数名 K -> queue_size, m -> momentum
                if dhn_cfg is not None:
                    normalized_cfg = {}
                    for k, v in dhn_cfg.items():
                        if k == 'K':
                            normalized_cfg['queue_size'] = v
                        elif k == 'm':
                            normalized_cfg['momentum'] = v
                        else:
                            normalized_cfg[k] = v
                    self.dhn_sampler = DHNSampler(dim=proj_dim, **normalized_cfg)
                else:
                    self.dhn_sampler = DHNSampler(dim=proj_dim)
            except Exception as e:
                print(f"[WARN] DHN import failed: {e}")
                self.dhn_sampler = None
                self.use_dhn = False

    def forward(self, z_vis, z_ir):
        """Forward pass for contrastive learning.

        Args:
            z_vis (Tensor): Visual domain features, shape (N, C).
            z_ir (Tensor): Infrared domain features, shape (N, C).

        Returns:
            dict: Loss dictionary containing 'loss_macl'.
        """
        # 输入检查
        if torch.isnan(z_vis).any() or torch.isnan(z_ir).any():
            print(f"[ERROR] NaN detected in MACL input! vis={torch.isnan(z_vis).sum()}, ir={torch.isnan(z_ir).sum()}")
            return {'loss_macl': torch.tensor(0.0, device=z_vis.device, requires_grad=True)}
        
        # L2归一化投影特征
        z_vis = F.normalize(self.proj(z_vis), dim=1)
        z_ir = F.normalize(self.proj(z_ir), dim=1)
        
        # 缓存 embeddings 用于 t-SNE 可视化
        self.latest_embeddings = {
            'vis': z_vis.detach().clone(),
            'ir': z_ir.detach().clone()
        }

        # 追加到环形缓冲，供一个 epoch 内聚合采样
        try:
            with torch.no_grad():
                vis_cpu = z_vis.detach().cpu()
                ir_cpu = z_ir.detach().cpu()
                if self._embed_buf_vis is None:
                    self._embed_buf_vis = vis_cpu
                else:
                    self._embed_buf_vis = torch.cat([self._embed_buf_vis, vis_cpu], dim=0)
                if self._embed_buf_ir is None:
                    self._embed_buf_ir = ir_cpu
                else:
                    self._embed_buf_ir = torch.cat([self._embed_buf_ir, ir_cpu], dim=0)
                # 截断到最大长度（保留最新样本）
                if self._embed_buf_vis.size(0) > self._embed_buf_max:
                    self._embed_buf_vis = self._embed_buf_vis[-self._embed_buf_max:]
                if self._embed_buf_ir.size(0) > self._embed_buf_max:
                    self._embed_buf_ir = self._embed_buf_ir[-self._embed_buf_max:]
        except Exception:
            pass  # 缓存失败不影响训练

    # 计算相似度矩阵并添加数值保护
        # sim = z_vis @ z_ir.T / tau
        sim = torch.matmul(z_vis, z_ir.t()) / (self.tau + 1e-8)
        # Clamp 防止极端值
        sim = torch.clamp(sim, min=-50.0, max=50.0)
        # 替换任何残留的 NaN/Inf
        sim = torch.nan_to_num(sim, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # 每100次迭代打印监控信息
        self.iter_count += 1
        if self.iter_count % 100 == 0:
            print(f"[DEBUG] MACL iter={self.iter_count}: mean(sim)={sim.mean():.4f}, "
                  f"std(sim)={sim.std():.4f}, tau={self.tau.item():.4f}")
        
        # 计算正样本相似度（对角线）
        pos_sim = torch.diag(sim)
        
        # 构建mask：对角线为0，其余为1（负样本）
        mask = torch.eye(len(z_vis), device=z_vis.device)
        
        # InfoNCE loss: -log(exp(pos) / sum(exp(all_negatives)))
        # 分子：exp(正样本相似度)
        exp_pos = torch.exp(pos_sim)
        # 分母：所有负样本的 exp 之和（排除对角线）+ 1e-8 防止除零
        exp_neg_sum = (torch.exp(sim) * (1 - mask)).sum(dim=1) + 1e-8
        
        # Loss
        loss_pos = -torch.log(exp_pos / (exp_neg_sum + exp_pos) + 1e-8).mean()

        # 若启用 DHN，添加困难负样本
        if self.use_dhn and self.dhn_sampler is not None:
            self.dhn_sampler.update_queue(z_ir.detach())
            hard_neg = self.dhn_sampler.sample_hard_negatives(z_vis)
            neg_sim = torch.sum(z_vis.unsqueeze(1) * hard_neg, dim=-1) / (self.tau + 1e-8)
            neg_sim = torch.clamp(neg_sim, min=-50.0, max=50.0)
            loss_neg = -torch.log(1 - torch.sigmoid(neg_sim) + 1e-8).mean()
        else:
            loss_neg = 0.0

        final_loss = loss_pos + 0.1 * loss_neg

        # 计算域对齐 MMD 损失（RBF kernel），对小 batch 做保护
        loss_domain = None
        if self.enable_domain_loss and z_vis.shape[0] >= 2 and z_ir.shape[0] >= 2:
            # MMD 计算使用 fp32 提高数值稳定性
            a = z_vis.float()
            b = z_ir.float()

            def pdist2(x, y):
                # ||x - y||^2 = x^2 + y^2 - 2xy
                x2 = (x * x).sum(dim=1, keepdim=True)
                y2 = (y * y).sum(dim=1, keepdim=True).t()
                xy = x @ y.t()
                d2 = torch.clamp(x2 + y2 - 2 * xy, min=0.0)
                return d2

            d_aa = pdist2(a, a)
            d_bb = pdist2(b, b)
            d_ab = pdist2(a, b)

            mmd = 0.0
            for s in self.mmd_kernels:
                gamma = 1.0 / (2.0 * (s ** 2) + 1e-12)
                k_aa = torch.exp(-gamma * d_aa)
                k_bb = torch.exp(-gamma * d_bb)
                k_ab = torch.exp(-gamma * d_ab)
                # 排除对角线对自核的影响（无偏估计近似）
                n = a.size(0)
                m = b.size(0)
                if n > 1:
                    mmd += (k_aa.sum() - k_aa.diag().sum()) / (n * (n - 1) + 1e-12)
                if m > 1:
                    mmd += (k_bb.sum() - k_bb.diag().sum()) / (m * (m - 1) + 1e-12)
                mmd -= 2.0 * k_ab.mean()
            mmd = mmd / max(len(self.mmd_kernels), 1)
            # 数值保护
            if torch.isnan(mmd) or torch.isinf(mmd):
                loss_domain = torch.tensor(0.0, device=a.device, requires_grad=True)
            else:
                loss_domain = mmd
                final_loss = final_loss + self.domain_weight * loss_domain
        else:
            if self.enable_domain_loss:
                loss_domain = torch.tensor(0.0, device=z_vis.device, requires_grad=True)
        
        # 最终安全检查
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"[ERROR] NaN/Inf in final MACL loss! Returning zero.")
            out = {'loss_macl': torch.tensor(0.0, device=z_vis.device, requires_grad=True)}
            if loss_domain is not None:
                out['loss_domain'] = torch.tensor(0.0, device=z_vis.device, requires_grad=True)
            return out

        out = {'loss_macl': final_loss}
        if loss_domain is not None:
            out['loss_domain'] = loss_domain
        return out

    # 供外部 Hook 使用：获取并清空缓冲的 embeddings
    def pop_buffer_embeddings(self):
        vis, ir = self._embed_buf_vis, self._embed_buf_ir
        self._embed_buf_vis, self._embed_buf_ir = None, None
        if vis is None or ir is None:
            return None
        return {
            'vis': vis.clone(),
            'ir': ir.clone()
        }
    
    def compute_loss(self, *args, **kwargs):
        """Placeholder loss computation method for compatibility.
        
        This method is called from StandardRoIHead when use_macl=True.
        Actual loss computation should be implemented based on bbox_results
        and img_metas from the forward pass.
        
        Returns:
            dict: Empty dict as placeholder. Override for actual implementation.
        """
        # 占位实现 - 避免训练时找不到方法报错
        # 实际使用时应根据 bbox_results 和特征提取 modal-aware features
        return {}
