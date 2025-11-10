"""
Gradient Flow Check & Visualization

- 前向+反向跑 1 个 batch（若数据集不可用则跳过，仅做挂钩演示）
- 统计每层梯度范数，绘制条形图至 logs/grad_flow_stageX.png
- 输出关键统计到 logs/research_validation_report.txt
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Any, Dict
import traceback


def safe_imports():
    try:
        import torch  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        from mmengine.config import Config  # noqa: F401
        from mmengine.runner.checkpoint import load_checkpoint  # noqa: F401
        from mmdet.registry import MODELS, DATASETS  # noqa: F401
        from mmengine.dataset import default_collate  # noqa: F401
        # 显式导入 transforms 以确保 PackDetInputs 等注册
        try:
            import mmdet.datasets.transforms  # noqa: F401
        except Exception:
            pass
        # 设置默认作用域，启用 mmdet 注册表查找
        try:
            from mmdet.utils import register_all_modules
            register_all_modules(init_default_scope=True)
        except Exception:
            pass
        return True, None
    except Exception as e:
        return False, e


def print_and_log(report_path: Path, text: str):
    print(text)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def collect_grads(model) -> List[Tuple[str, float]]:
    if hasattr(model, 'module'):
        model = model.module
    grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads.append((name, p.grad.detach().data.norm(2).item()))
    return grads


def plot_grad_flow(pairs: List[Tuple[str, float]], out_path: Path):
    import matplotlib.pyplot as plt
    if not pairs:
        return False
    names = [n for n, _ in pairs]
    values = [v for _, v in pairs]
    # 限制最多显示前 100 项，避免图过大
    names = names[:100]
    values = values[:100]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.title('Gradient L2 Norm (Top 100)')
    plt.xlabel('Param Index')
    plt.ylabel('L2 Norm')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _patch_data_for_loss(model, inputs, data_samples, prefer_multimodal: bool = True):
    """Best-effort patch to ensure inputs and data_samples meet 3.x contracts.

    - Convert inputs uint8 -> float32 in [0,1]
    - Ensure inputs on same device as model
    - Inject common metainfo (pad_shape/img_shape/ori_shape/scale_factor)
    - Ensure gt_instances.bboxes is a BaseBoxes subclass (HorizontalBoxes)

    prefer_multimodal: when True, single Tensor inputs will be duplicated into
    a dict with keys {'visible','infrared'} for models expecting dual-modality.
    When False (e.g., Stage1), keep Tensor as-is.
    """
    import torch
    from mmdet.structures import DetDataSample
    try:
        from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes  # type: ignore
    except Exception:
        HorizontalBoxes = None
        BaseBoxes = None

    mdl_dev = next(model.parameters()).device

    # inputs: support Tensor | list[Tensor] | dict[str, Tensor|list[Tensor]]
    def _prep_tensor(t):
        if isinstance(t, list):
            try:
                t = torch.stack(t, dim=0)
            except Exception:
                pass
        if hasattr(t, 'to'):
            if getattr(t, 'dtype', None) and str(t.dtype) in ('torch.uint8', 'torch.int8'):
                t = t.float().div_(255.0)
            t = t.to(mdl_dev)
        return t

    # Normalize input structure to expected multimodal dict if needed
    if isinstance(inputs, dict):
        for k, v in list(inputs.items()):
            inputs[k] = _prep_tensor(v)
        # Ensure both modalities exist if model expects them (heuristic)
        # If only one modality present, duplicate to missing keys to unblock forward
        if ('infrared' in inputs) and ('visible' not in inputs):
            inputs['visible'] = inputs['infrared']
        if ('visible' in inputs) and ('infrared' not in inputs):
            inputs['infrared'] = inputs['visible']
    else:
        # Single tensor
        t = _prep_tensor(inputs)
        if prefer_multimodal:
            inputs = {'visible': t, 'infrared': t}
        else:
            inputs = t

    def _ensure_sample(ds: DetDataSample):
        # meta
        # infer H, W from a tensor among inputs
        def _infer_shape(inp):
            if hasattr(inp, 'shape') and len(inp.shape) >= 4:
                return int(inp.shape[2]), int(inp.shape[3])
            return None, None
        H, W = None, None
        if isinstance(inputs, dict):
            for _v in inputs.values():
                H, W = _infer_shape(_v)
                if H and W:
                    break
        else:
            H, W = _infer_shape(inputs)
        shape3 = (int(H), int(W), 3) if H and W else (640, 640, 3)
        meta_defaults = {
            'pad_shape': shape3,
            'img_shape': shape3,
            'ori_shape': shape3,
            'scale_factor': (1.0, 1.0, 1.0, 1.0)
        }
        mi = getattr(ds, 'metainfo', {})
        missing = {k: v for k, v in meta_defaults.items() if k not in mi}
        if missing:
            try:
                ds.set_metainfo(missing)
            except Exception:
                pass
        # gt instances
        inst = getattr(ds, 'gt_instances', None)
        if inst is not None:
            # bboxes -> BaseBoxes
            b = getattr(inst, 'bboxes', None)
            if b is not None:
                if HorizontalBoxes is not None:
                    try:
                        # if already BaseBoxes skip
                        is_base = False
                        if BaseBoxes is not None:
                            is_base = isinstance(b, BaseBoxes)
                        if not is_base:
                            if hasattr(b, 'to'):
                                b = b.to(mdl_dev)
                            # ensure float
                            try:
                                import torch as _t
                                if isinstance(b, _t.Tensor) and b.dtype not in (_t.float32, _t.float64):
                                    b = b.float()
                            except Exception:
                                pass
                            b = HorizontalBoxes(b)
                            inst.bboxes = b
                    except Exception:
                        pass
            # labels
            if not hasattr(inst, 'labels') or getattr(inst, 'labels', None) is None:
                try:
                    import torch as _t
                    n = len(getattr(inst, 'bboxes', [])) if hasattr(inst, 'bboxes') else 0
                    inst.labels = _t.zeros((n,), dtype=_t.long, device=mdl_dev)
                except Exception:
                    pass
        return ds

    if data_samples is not None:
        if isinstance(data_samples, list):
            data_samples = [_ensure_sample(ds) for ds in data_samples]
        else:
            data_samples = _ensure_sample(data_samples)

    return inputs, data_samples


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _ensure_full_model(cfg, report_path: Path):
    required = {'backbone', 'rpn_head', 'train_cfg', 'test_cfg'}
    model_dict = cfg.model
    missing = [k for k in required if k not in model_dict]
    if not missing:
        return cfg
    try:
        from mmengine.config import Config
        base_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
        base_model = base_cfg.get('model', {})
        merged = _deep_update(base_model, model_dict)
        cfg.model = merged
        print_and_log(report_path, f"[INFO] 已自动合并基础模型字段，补齐缺失键：{missing}")
    except Exception as e:
        print_and_log(report_path, f"[WARN] 无法自动合并基础模型：{e}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description='梯度流可视化检查')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--stage', type=str, default='auto', choices=['auto', 'stage1', 'stage2', 'stage3'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-root', type=str, default=None, help='覆盖 train 数据根路径')
    parser.add_argument('--llvip-root', type=str, default='C:/LLVIP/LLVIP')
    parser.add_argument('--kaist-root', type=str, default='C:/KAIST_processed')
    parser.add_argument('--m3fd-root', type=str, default='C:/M3FD')
    args = parser.parse_args()

    ok, err = safe_imports()
    report_path = Path('logs/research_validation_report.txt')
    if not ok:
        print_and_log(report_path, f'[ERROR] 依赖缺失：{err}')
        raise SystemExit(1)

    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS, DATASETS
    # 强制导入 transforms 以注册 PackDetInputs（有些环境懒加载可能失败）
    try:
        import mmdet.datasets  # noqa: F401
        import mmdet.datasets.transforms as _t  # noqa: F401
        use_v3 = True
    except Exception as e:
        print(f"[WARN] transforms 导入失败（可能为 MMDet 2.x 环境）：{e}")
        use_v3 = False
    import torch
    from torch.utils.data import DataLoader
    try:
        from mmengine.dataset import default_collate
    except Exception:
        from torch.utils.data import default_collate  # type: ignore

    stage = args.stage
    if stage == 'auto':
        low = args.config.lower()
        if 'stage1' in low:
            stage = 'stage1'
        elif 'stage2' in low:
            stage = 'stage2'
        else:
            stage = 'stage3'

    cfg = Config.fromfile(args.config)
    cfg = _ensure_full_model(cfg, report_path)
    # Ensure data_preprocessor exists to handle device transfer and stacking
    if 'data_preprocessor' not in cfg.model:
        cfg.model['data_preprocessor'] = dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32
        )
    model = MODELS.build(cfg.model)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    model.to(device)
    model.train()

    # 可选加载权重
    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            load_checkpoint(model, args.checkpoint, map_location='cpu')
        except Exception as e:
            print_and_log(report_path, f"[WARN] 加载权重失败：{e}")
    elif args.checkpoint:
        print_and_log(report_path, f"[WARN] 权重文件不存在，跳过加载：{args.checkpoint}")

    # === 构建数据集与 DataLoader，跑 1 个 batch ===
    grads = []
    loss_value = None
    ran_batch = False
    # 兼容 mmdet 2.x/3.x 的数据键
    def _get_train_dataset_cfg(_cfg):
        if hasattr(_cfg, 'train_dataloader') and isinstance(_cfg.train_dataloader, dict):
            return _cfg.train_dataloader.get('dataset', None), 'v3'
        # legacy
        if hasattr(_cfg, 'data') and isinstance(_cfg.data, dict) and 'train' in _cfg.data:
            return _cfg.data['train'], 'v2'
        return None, 'na'
    def _get_fallback_dataset_cfg(_cfg):
        # try test/val dataset for a supervised batch if train missing pipeline
        if hasattr(_cfg, 'test_dataloader') and isinstance(_cfg.test_dataloader, dict):
            ds = _cfg.test_dataloader.get('dataset', None)
            if ds is not None:
                return ds, 'v3_test'
        if hasattr(_cfg, 'val_dataloader') and isinstance(_cfg.val_dataloader, dict):
            ds = _cfg.val_dataloader.get('dataset', None)
            if ds is not None:
                return ds, 'v3_val'
        if hasattr(_cfg, 'data') and isinstance(_cfg.data, dict):
            for k in ('test', 'val'):
                if k in _cfg.data:
                    return _cfg.data[k], f'v2_{k}'
        return None, 'na'

    ds_cfg, ds_ver = _get_train_dataset_cfg(cfg)
    if ds_cfg is None or not isinstance(ds_cfg, dict) or 'pipeline' not in ds_cfg:
        if ds_cfg is not None and 'pipeline' not in ds_cfg:
            print_and_log(report_path, '[INFO] 训练集缺少 pipeline，尝试回退到 test/val 数据集用于单 batch 检查。')
        ds_cfg, ds_ver = _get_fallback_dataset_cfg(cfg)
    if ds_cfg is None:
        print_and_log(report_path, '[WARN] 未在配置中找到 train 数据集定义，跳过实测前反向。')
    else:
        # 若仍无 pipeline，写入最小 pipeline 保证可构建（兼容 2.x/3.x）
        def _ensure_pipeline(node: Dict):
            if 'pipeline' not in node or not node.get('pipeline'):
                if use_v3:
                    node['pipeline'] = [
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','modality'))
                    ]
                else:
                    node['pipeline'] = [
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations'),
                        dict(type='DefaultFormatBundle'),
                        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                    ]
            return node
        # 尝试覆盖 data_root
        def _override_root(node: Dict):
            if 'data_root' in node and args.data_root:
                node['data_root'] = args.data_root
            # 常见根路径自动修正
            if 'data_root' in node and not args.data_root:
                root = node['data_root']
                if 'KAIST' in root or 'kaist' in root.lower():
                    node['data_root'] = args.kaist_root
                elif 'LLVIP' in root or 'llvip' in root.lower():
                    node['data_root'] = args.llvip_root
                elif 'M3FD' in root or 'm3fd' in root.lower():
                    node['data_root'] = args.m3fd_root
            # 兼容 data_prefix.sub_data_root
            if 'data_prefix' in node and isinstance(node['data_prefix'], dict):
                if 'sub_data_root' in node['data_prefix']:
                    if args.data_root:
                        node['data_prefix']['sub_data_root'] = args.data_root
                    else:
                        # 和 data_root 保持一致
                        node['data_prefix']['sub_data_root'] = node.get('data_root', node['data_prefix']['sub_data_root'])
            # 递归处理 ConcatDataset 等嵌套结构
            if node.get('type') == 'ConcatDataset' and isinstance(node.get('datasets', None), list):
                for i, sub in enumerate(node['datasets']):
                    node['datasets'][i] = _override_root(_ensure_pipeline(sub))
            return node

        try:
            ds_cfg = _ensure_pipeline(ds_cfg.copy() if isinstance(ds_cfg, dict) else dict(ds_cfg))
            ds_cfg = _override_root(ds_cfg)
            dataset = DATASETS.build(ds_cfg)
            # ==== 调试：打印 dataset[0] 结构 ====
            try:
                sample0 = dataset[0]
                def _summarize(obj, prefix='sample0'):
                    if isinstance(obj, dict):
                        keys = list(obj.keys())
                        print(f'[DEBUG] {prefix} dict keys={keys}')
                        for k in keys:
                            v = obj[k]
                            if hasattr(v, 'shape'):
                                print(f'[DEBUG]   {k}: type={type(v)} shape={getattr(v,"shape",None)} dtype={getattr(v,"dtype",None)}')
                            else:
                                print(f'[DEBUG]   {k}: type={type(v)}')
                    else:
                        print(f'[DEBUG] {prefix} type={type(obj)} attrs={dir(obj)[:10]}')
                _summarize(sample0)
            except Exception as e_dbg:
                print(f'[DEBUG] 读取 dataset[0] 失败: {e_dbg}')
            batch_size = args.batch_size
            num_workers = args.num_workers
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=default_collate)
            for batch in loader:
                # ==== 调试：打印 batch 结构摘要 ====
                try:
                    if isinstance(batch, dict):
                        print('[DEBUG] batch dict keys:', list(batch.keys()))
                        if 'inputs' in batch and hasattr(batch['inputs'], 'shape'):
                            print('[DEBUG] batch.inputs shape:', batch['inputs'].shape, 'dtype:', getattr(batch['inputs'],'dtype',None))
                    elif isinstance(batch, list):
                        print('[DEBUG] batch is list len=', len(batch))
                    else:
                        print('[DEBUG] batch type=', type(batch))
                except Exception as e_bdbg:
                    print('[DEBUG] 打印 batch 信息失败:', e_bdbg)
                # mmdet 3.x: batch 是包含 'inputs' 与 'data_samples' 的 dict
                if isinstance(batch, dict) and 'inputs' in batch:
                    inputs = batch['inputs']
                    data_samples = batch.get('data_samples', None)
                    # 如果 inputs 是 list[Tensor]，stack 为 batched tensor
                    if isinstance(inputs, list):
                        try:
                            import torch as _torch
                            inputs = _torch.stack(inputs, dim=0)
                        except Exception:
                            pass
                    # 尽量与模型设备一致
                    try:
                        # 统一采用多模态输入适配，避免模型内部将Tensor包裹为dict导致设备不一致
                        inputs, data_samples = _patch_data_for_loss(model, inputs, data_samples, prefer_multimodal=True)
                    except Exception:
                        pass
                    # 强制设备对齐（字典/张量）
                    try:
                        import torch as _torch
                        if isinstance(inputs, dict):
                            for _k, _v in list(inputs.items()):
                                if hasattr(_v, 'to'):
                                    inputs[_k] = _v.to(device)
                        elif hasattr(inputs, 'to'):
                            inputs = inputs.to(device)
                    except Exception:
                        pass
                    try:
                        losses = model(inputs, data_samples, mode='loss')
                    except TypeError:
                        # 兼容老式 forward_train(dict)
                        losses = model.forward_train(**batch) if hasattr(model, 'forward_train') else model(**batch)
                    except KeyError as ke:
                        # 补齐常见缺失的 meta key（如 pad_shape）再重试一次
                        missing_key = str(ke).strip("'")
                        if data_samples is not None:
                            def _inject(ds):
                                if not hasattr(ds, 'metainfo'):
                                    return
                                mi = ds.metainfo
                                h = getattr(inputs, 'shape', [None, None, None, None])[2]
                                w = getattr(inputs, 'shape', [None, None, None, None])[3]
                                shape3 = (h, w, 3) if h and w else (640, 640, 3)
                                if 'pad_shape' not in mi:
                                    ds.set_metainfo(dict(pad_shape=shape3))
                                if 'ori_shape' not in mi:
                                    ds.set_metainfo(dict(ori_shape=shape3))
                                if 'img_shape' not in mi:
                                    ds.set_metainfo(dict(img_shape=shape3))
                            if isinstance(data_samples, list):
                                for _ds in data_samples:
                                    _inject(_ds)
                            else:
                                _inject(data_samples)
                        try:
                            losses = model(inputs, data_samples, mode='loss')
                        except Exception as e2:
                            print('[DEBUG] 二次前向仍失败 (KeyError 补齐后):', e2)
                            traceback.print_exc()
                            raise e2
                else:
                    # 尝试直接解包到 forward_train
                    losses = model.forward_train(**batch) if hasattr(model, 'forward_train') else model(**batch)

                # 聚合 loss
                total_loss = 0.0
                if isinstance(losses, dict):
                    for k, v in losses.items():
                        if hasattr(v, 'mean'):
                            total_loss += v.mean()
                elif hasattr(losses, 'mean'):
                    total_loss = losses.mean()
                else:
                    total_loss = torch.as_tensor(losses, device=device).mean()

                total_loss.backward()
                ran_batch = True
                loss_value = float(total_loss.detach().cpu())
                break  # 仅 1 个 batch
        except Exception as e:
            print_and_log(report_path, f'[WARN] 实测 1 个 batch 前反向失败：{e}')
            traceback.print_exc()

    # 收集已有梯度（若上一段跑通）
    try:
        grads = collect_grads(model)
    except Exception:
        grads = []

    out_png = Path(f'logs/grad_flow_{stage}.png')
    if plot_grad_flow(grads, out_png):
        print_and_log(report_path, f'[OK] 梯度图已保存：{out_png}')
    else:
        print_and_log(report_path, '[WARN] 未能生成梯度图（可能未进行反向传播或无梯度）')

    # 基础统计
    if grads:
        nz = sum(1 for _, v in grads if v > 0)
        print_and_log(report_path, f'[STATS] 有梯度参数数：{nz} / {len(grads)}')
    if ran_batch:
        print_and_log(report_path, f'[OK] 实测 1 batch 完成，loss={loss_value:.6f}')
    else:
        print_and_log(report_path, '[INFO] 未跑实测 batch（可能因数据集未配置或构建失败）')

    print_and_log(report_path, '梯度流检查完成。')


if __name__ == '__main__':
    main()
