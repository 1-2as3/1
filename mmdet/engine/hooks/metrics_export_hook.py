"""
科研级 MetricsExportHook：
- 自动绘制 loss_macl / loss_total 曲线与梯度范数热力图
- 生成交互式 HTML 报告
- 多实验对比
- 防 NaN 保护
- 自动保存最佳 checkpoint（优先 mAP，其次最低 loss_total）
- Windows 安全 I/O
"""
import os
import csv
import json
import math
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

# 安全使用 matplotlib（非交互式后端，避免 Windows/无显示环境问题）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# 可选：交互式 HTML 报告
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def _atomic_write(path: str, data_bytes: bytes) -> None:
    """Windows 安全的原子写文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        f.write(data_bytes)
    os.replace(tmp_path, path)


@HOOKS.register_module()
class MetricsExportHook(Hook):
    """科研级监控 Hook + HTML 报告版"""

    priority = 'NORMAL'

    def __init__(self,
                 out_dir: str = 'work_dirs/metrics_logs',
                 interval: int = 1,
                 clip_grad_max_norm: float = 5.0,
                 record_grad_norm: bool = True,
                 enable_html_report: bool = True,
                 enable_multi_exp_compare: bool = True,
                 sync_best_ckpt_to_workdir: bool = True) -> None:
        # 新增 run_id 机制：自动按时间分组每次运行
        self.run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.out_dir = os.path.join(out_dir, self.run_id)
        self.sync_best_ckpt_to_workdir = sync_best_ckpt_to_workdir
        
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[MetricsExportHook] Created new run folder: {self.out_dir}")
        
        self.interval = interval
        self.clip_grad_max_norm = clip_grad_max_norm
        self.record_grad_norm = record_grad_norm
        self.enable_html_report = enable_html_report
        self.enable_multi_exp_compare = enable_multi_exp_compare

        self.loss_history: List[Dict[str, float]] = []
        self.grad_norms: List[float] = []
        self.best_mAP: float = -1.0
        self.best_loss: float = float('inf')

    # --------- 每 iter：收集指标、梯度范数与裁剪 ---------
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[dict] = None):
        if outputs is None:
            # 回退到 message_hub
            log_scalars = getattr(runner.message_hub, 'log_scalars', {}) or {}
            safe_losses: Dict[str, float] = {}
            for k, v in log_scalars.items():
                if isinstance(v, dict) and 'data' in v:
                    val = v['data']
                    if torch.is_tensor(val):
                        val = val.detach()
                        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                        safe_losses[k] = float(val.item())
                    elif isinstance(val, (int, float)):
                        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                            val = 0.0
                        safe_losses[k] = float(val)
            if safe_losses:
                self.loss_history.append(safe_losses)
        else:
            safe_losses: Dict[str, float] = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    val = torch.nan_to_num(v.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                    safe_losses[k] = float(val.item())
                elif isinstance(v, list):
                    vals = []
                    for x in v:
                        if torch.is_tensor(x):
                            x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                            vals.append(float(x.item()))
                        elif isinstance(x, (int, float)):
                            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                                x = 0.0
                            vals.append(float(x))
                    if len(vals) > 0:
                        safe_losses[k] = float(np.mean(vals))
                elif isinstance(v, (int, float)):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        v = 0.0
                    safe_losses[k] = float(v)
            if safe_losses:
                self.loss_history.append(safe_losses)

        # 梯度范数统计与裁剪
        if self.record_grad_norm and hasattr(runner, 'model') and runner.model is not None:
            total_sq = 0.0
            for p in runner.model.parameters():
                if p is not None and p.grad is not None:
                    # 使用 FP32 计算范数，规避溢出
                    param_norm = p.grad.data.detach().float().norm(2)
                    total_sq += float(param_norm.item() ** 2)
            total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
            self.grad_norms.append(total_norm)
            # 裁剪（无害冗余，若 OptimWrapper 也裁剪则数值更稳健）
            try:
                torch.nn.utils.clip_grad_norm_(runner.model.parameters(), max_norm=self.clip_grad_max_norm)
            except Exception:
                pass

    # --------- 每 epoch：导出文件、生成报告、保存最佳 ---------
    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        if epoch % max(1, int(self.interval)) != 0:
            return

        # CSV 导出（Windows 安全 I/O）
        csv_path = os.path.join(self.out_dir, 'metrics_log.csv')
        keys = sorted(set(k for d in self.loss_history for k in d.keys()))
        header = ['epoch_iter'] + keys + (['grad_norm'] if self.record_grad_norm else [])
        rows: List[Dict[str, Any]] = []
        for i, d in enumerate(self.loss_history):
            row = {'epoch_iter': i}
            row.update({k: d.get(k, 0.0) for k in keys})
            if self.record_grad_norm:
                row['grad_norm'] = self.grad_norms[i] if i < len(self.grad_norms) else None
            rows.append(row)
        _atomic_write(csv_path, self._to_csv_bytes(header, rows))

        # 绘制静态图：loss_total / loss_macl
        if HAS_MATPLOTLIB:
            try:
                plt.figure(figsize=(8, 5))
                plt.plot([x.get('loss_total', 0.0) for x in self.loss_history], label='loss_total', lw=2)
                plt.plot([x.get('loss_macl', 0.0) for x in self.loss_history], label='loss_macl', lw=2, linestyle='--')
                plt.legend(); plt.xlabel('Iteration'); plt.ylabel('Loss')
                plt.title(f'Loss Curve (Epoch {epoch})')
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f'loss_curve_epoch{epoch}.png'), dpi=200)
                plt.close()
            except Exception:
                pass

            # 梯度范数热力图 PNG
            if self.record_grad_norm and len(self.grad_norms) > 0:
                try:
                    plt.figure(figsize=(10, 1.8))
                    gn = np.array(self.grad_norms, dtype=float)[None, :]
                    plt.imshow(gn, aspect='auto', cmap='plasma')
                    plt.colorbar(label='grad_norm', fraction=0.046, pad=0.04)
                    plt.yticks([])
                    plt.xlabel('Iteration')
                    plt.title(f'Grad Norm Heatmap (Epoch {epoch})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, f'grad_norm_heatmap_epoch{epoch}.png'), dpi=200)
                    plt.close()
                except Exception:
                    pass

        # 记录并同步最佳模型：优先使用 coco/bbox_mAP；否则按最低 loss_total
        current_map = self._read_current_map(runner)
        current_loss_total = self.loss_history[-1].get('loss_total', None) if self.loss_history else None
        best_saved = False
        ckpt_name = None
        
        if current_map is not None:
            if current_map > self.best_mAP:
                self.best_mAP = current_map
                ckpt_name = f'best_epoch{epoch}_mAP{current_map:.3f}.pth'
                ckpt_path = os.path.join(self.out_dir, ckpt_name)
                self._save_checkpoint(runner, ckpt_path)
                best_saved = True
        elif isinstance(current_loss_total, (int, float)):
            if current_loss_total < self.best_loss:
                self.best_loss = float(current_loss_total)
                ckpt_name = f'best_epoch{epoch}_loss{self.best_loss:.4f}.pth'
                ckpt_path = os.path.join(self.out_dir, ckpt_name)
                self._save_checkpoint(runner, ckpt_path)
                best_saved = True
        
        # 同步最佳 checkpoint 到主 work_dir（与 MMEngine 默认断点合流）
        if best_saved and ckpt_name and self.sync_best_ckpt_to_workdir:
            if hasattr(runner, 'work_dir') and runner.work_dir:
                try:
                    dst = os.path.join(runner.work_dir, ckpt_name)
                    shutil.copy(ckpt_path, dst)
                    runner.logger.info(f'[OK] Synced best checkpoint to work_dir: {dst}')
                except Exception as e:
                    runner.logger.warning(f'[WARN] Failed to sync checkpoint: {e}')

        # 生成交互式 HTML 报告
        if self.enable_html_report and HAS_PLOTLY:
            try:
                self._generate_html_report(epoch)
                runner.logger.info('[OK] HTML report generated')
            except Exception:
                pass

        if best_saved and not (self.sync_best_ckpt_to_workdir and hasattr(runner, 'work_dir')):
            # 只在未同步时显示保存消息（避免重复日志）
            runner.logger.info('[OK] Best checkpoint saved')

        # 多实验对比
        if self.enable_multi_exp_compare:
            try:
                self._compare_experiments()
            except Exception:
                pass

        # === 1️⃣ 保存配置文件快照 ===
        if hasattr(runner, 'cfg') and hasattr(runner.cfg, 'filename'):
            cfg_path = runner.cfg.filename
            if cfg_path and os.path.isfile(cfg_path):
                dst_cfg = os.path.join(self.out_dir, f'config_snapshot_epoch{epoch}.py')
                try:
                    shutil.copy(cfg_path, dst_cfg)
                    runner.logger.info(f'[OK] Config snapshot saved: {os.path.basename(dst_cfg)}')
                except Exception as e:
                    runner.logger.warning(f'[WARN] Config snapshot failed: {e}')

        # === 2️⃣ TensorBoard 日志同步 ===
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(runner.work_dir, 'tensorboard_logs')
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            
            # 写入所有历史数据
            loss_total = [x.get('loss_total', 0.0) for x in self.loss_history]
            loss_macl = [x.get('loss_macl', 0.0) for x in self.loss_history]
            
            for i in range(len(loss_total)):
                writer.add_scalar('Loss/total', loss_total[i], i)
                if i < len(loss_macl) and loss_macl[i] != 0.0:
                    writer.add_scalar('Loss/macl', loss_macl[i], i)
                if self.record_grad_norm and i < len(self.grad_norms):
                    writer.add_scalar('Grad/norm', self.grad_norms[i], i)
            
            # 写入 epoch 级别的指标
            writer.add_scalar('Epoch/best_loss', self.best_loss, epoch)
            if self.best_mAP > -1.0:
                writer.add_scalar('Epoch/best_mAP', self.best_mAP, epoch)
            
            writer.flush()
            writer.close()
            runner.logger.info(f'[OK] TensorBoard logs updated: {tb_dir}')
        except ImportError:
            runner.logger.warning('[WARN] TensorBoard not available (pip install tensorboard)')
        except Exception as e:
            runner.logger.warning(f'[WARN] TensorBoard write failed: {e}')

    # --------- 小功能：HTML 报告与多实验对比 ---------
    def _generate_html_report(self, epoch: int) -> None:
        loss_total = [x.get('loss_total', 0.0) for x in self.loss_history]
        loss_macl = [x.get('loss_macl', 0.0) for x in self.loss_history]
        grad_norm = self.grad_norms

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_total, name='loss_total'))
        fig.add_trace(go.Scatter(y=loss_macl, name='loss_macl'))
        fig.update_layout(title='Loss Trend', xaxis_title='Iter', yaxis_title='Loss')

        fig2 = go.Figure(data=go.Heatmap(z=[grad_norm], colorscale='plasma', showscale=True))
        fig2.update_layout(title='Grad Norm Heatmap', xaxis_title='Iter', yaxis_title='')

        html_path = os.path.join(self.out_dir, f'report_epoch{epoch}.html')
        html_parts = [
            '<h2>Training Report</h2>',
            f'<p>Epoch: {epoch}</p>',
            f'<p>Best mAP: {self.best_mAP:.3f}</p>',
            fig.to_html(full_html=False, include_plotlyjs='cdn'),
            fig2.to_html(full_html=False, include_plotlyjs=False)
        ]
        _atomic_write(html_path, '\n'.join(html_parts).encode('utf-8'))

    def _compare_experiments(self) -> None:
        base_dir = os.path.dirname(self.out_dir)
        if not os.path.isdir(base_dir):
            return
        exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        curves = []
        for exp in exp_dirs:
            path = os.path.join(base_dir, exp, 'metrics_log.csv')
            if os.path.exists(path):
                try:
                    data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8')
                    if hasattr(data, 'dtype') and 'loss_total' in data.dtype.names:
                        curves.append((exp, data['loss_total']))
                except Exception:
                    continue
        if len(curves) > 1 and HAS_PLOTLY:
            fig = go.Figure()
            for exp, vals in curves:
                fig.add_trace(go.Scatter(y=vals, name=exp))
            fig.update_layout(title='Cross-Experiment Loss Comparison', xaxis_title='Iteration', yaxis_title='loss_total')
            fig.write_html(os.path.join(self.out_dir, 'compare_experiments.html'))

    # --------- 工具方法 ---------
    def _to_csv_bytes(self, header: List[str], rows: List[Dict[str, Any]]) -> bytes:
        from io import StringIO
        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return sio.getvalue().encode('utf-8')

    def _read_current_map(self, runner) -> Optional[float]:
        mh = getattr(runner, 'message_hub', None)
        if mh is None:
            return None
        # 常见 key：'coco/bbox_mAP' 或 'bbox_mAP'
        candidates = ['coco/bbox_mAP', 'bbox_mAP', 'coco/bbox_mAP_50', 'mAP']
        # runtime_info 里可能有，也可能在 log_scalars 历史里
        try:
            for k in candidates:
                if hasattr(mh, 'runtime_info') and k in mh.runtime_info:
                    v = mh.runtime_info[k]
                    if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                        return float(v)
        except Exception:
            pass
        try:
            ls = getattr(mh, 'log_scalars', {}) or {}
            for k in candidates:
                if k in ls:
                    v = ls[k]
                    if isinstance(v, dict) and 'data' in v:
                        v = v['data']
                    if torch.is_tensor(v):
                        v = float(v.item())
                    if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                        return float(v)
        except Exception:
            pass
        return None

    def _save_checkpoint(self, runner, filepath: str) -> None:
        # 兼容 DP / DDP 包装
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        state_dict = model.state_dict()
        # 保存包含少量 meta 的字典，便于追溯
        meta = {
            'epoch': runner.epoch,
            'iter': runner.iter,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        payload = {'state_dict': state_dict, 'meta': meta}
        torch.save(payload, filepath)

    # --------- 训练结束 Hook ---------
    def after_train(self, runner) -> None:
        """训练结束后：压缩结果目录"""
        try:
            # 压缩当前 run 结果目录
            base_name = self.out_dir.rstrip(os.sep)
            zip_path = shutil.make_archive(base_name, 'zip', os.path.dirname(self.out_dir), os.path.basename(self.out_dir))
            runner.logger.info(f'[OK] Compressed run results: {zip_path}')
        except Exception as e:
            runner.logger.warning(f'[WARN] Compression failed: {e}')
        
        # 打印最终总结
        runner.logger.info('=' * 60)
        runner.logger.info('Training Completed - Summary:')
        runner.logger.info(f'  Run ID: {self.run_id}')
        runner.logger.info(f'  Output Dir: {self.out_dir}')
        runner.logger.info(f'  Best Loss: {self.best_loss:.4f}')
        if self.best_mAP > -1.0:
            runner.logger.info(f'  Best mAP: {self.best_mAP:.3f}')
        runner.logger.info(f'  Total Iterations: {len(self.loss_history)}')
        runner.logger.info('=' * 60)
