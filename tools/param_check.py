import argparse
from mmengine import Config
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Config file path')
    parser.add_argument('--ckpt', default=None, help='Checkpoint to load (optional)')
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model)
    model.init_weights()
    if args.ckpt:
        load_checkpoint(model, args.ckpt, map_location='cpu')

    total, trainable = 0, 0
    frozen_layers = []
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen_layers.append(name)

    print('[param_check] Total params: %.2fM' % (total/1e6))
    print('[param_check] Trainable params: %.2fM (%.2f%%)' % (trainable/1e6, 100*trainable/max(total,1)))
    if hasattr(cfg.model.get('backbone', {}), 'frozen_stages'):
        print('[param_check] frozen_stages:', cfg.model.backbone.frozen_stages)
    print('[param_check] Frozen tensors count:', len(frozen_layers))
    print('[param_check] Example frozen names (first 10):')
    for n in frozen_layers[:10]:
        print('  -', n)

if __name__ == '__main__':
    main()
"""
Parameter Trainability Check

- 加载指定 config 与可选 checkpoint
- 按模块统计可训练参数：backbone/neck/roi_head/macl/msp/dhn/man/others
- 输出到终端并写入 logs/research_validation_report.txt
- 支持 --stage 自动推断或手动指定
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Any


def safe_imports():
    try:
        import torch  # noqa: F401
        from mmengine.config import Config  # noqa: F401
        from mmengine.runner.checkpoint import load_checkpoint  # noqa: F401
        from mmdet.registry import MODELS  # noqa: F401
        return True, None
    except Exception as e:
        return False, e


def analyze_params(model) -> Dict[str, int]:
    counts = {
        'total': 0,
        'trainable': 0,
        'backbone': 0,
        'backbone_trainable': 0,
        'neck': 0,
        'neck_trainable': 0,
        'roi_head': 0,
        'roi_head_trainable': 0,
        'macl': 0,
        'macl_trainable': 0,
        'msp': 0,
        'msp_trainable': 0,
        'dhn': 0,
        'dhn_trainable': 0,
        'man': 0,
        'man_trainable': 0,
        'others': 0,
        'others_trainable': 0,
    }
    if hasattr(model, 'module'):
        model = model.module

    for name, p in model.named_parameters():
        n = p.numel()
        counts['total'] += n
        if p.requires_grad:
            counts['trainable'] += n
        key_base = None
        if name.startswith('backbone') or '.backbone.' in name:
            key_base = 'backbone'
        elif name.startswith('neck') or '.neck.' in name:
            key_base = 'neck'
        elif 'roi_head' in name:
            key_base = 'roi_head'
        elif 'macl' in name or 'contrast' in name:
            key_base = 'macl'
        elif 'msp' in name or 'modality_specific' in name:
            key_base = 'msp'
        elif 'dhn' in name or 'harmon' in name:
            key_base = 'dhn'
        elif 'modality_adaptive' in name or '.man.' in name or 'ModalityAdaptive' in name:
            key_base = 'man'
        else:
            key_base = 'others'
        counts[key_base] += n
        if p.requires_grad:
            counts[f'{key_base}_trainable'] += n
    return counts


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update base with override and return new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _ensure_full_model(cfg, report_path: Path):
    """Ensure cfg.model contains required sub-keys by merging base model if needed."""
    required = {'backbone', 'rpn_head', 'train_cfg', 'test_cfg'}
    model_dict = cfg.model
    missing = [k for k in required if k not in model_dict]
    if not missing:
        return cfg

    # Try load default base model and merge
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


def print_and_log(report_path: Path, text: str):
    print(text)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def main():
    parser = argparse.ArgumentParser(description='参数可训练性检查')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='可选：权重路径')
    parser.add_argument('--stage', type=str, default='auto', choices=['auto', 'stage1', 'stage2', 'stage3'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    ok, err = safe_imports()
    report_path = Path('logs/research_validation_report.txt')
    if not ok:
        print_and_log(report_path, f'[ERROR] 依赖缺失：{err}')
        print_and_log(report_path, '请先安装依赖并确保可导入 mmdet 与 torch.cuda 正常。')
        raise SystemExit(1)

    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS
    import torch

    stage = args.stage
    if stage == 'auto':
        low = args.config.lower()
        if 'stage1' in low:
            stage = 'stage1'
        elif 'stage2' in low:
            stage = 'stage2'
        else:
            stage = 'stage3'

    cfg_path = args.config
    checkpoint = args.checkpoint

    cfg = Config.fromfile(cfg_path)
    cfg = _ensure_full_model(cfg, report_path)

    # 构建模型而非使用 init_detector（避免对 backbone 键的假设）
    model = MODELS.build(cfg.model)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    model.to(device)
    model.train()

    # 加载权重（可选，若路径存在）
    if checkpoint and Path(checkpoint).exists():
        try:
            load_checkpoint(model, checkpoint, map_location='cpu')
        except Exception as e:
            print_and_log(report_path, f"[WARN] 加载权重失败：{e}")
    elif checkpoint:
        print_and_log(report_path, f"[WARN] 权重文件不存在，跳过加载：{checkpoint}")

    counts = analyze_params(model)

    ratio = counts['trainable'] / max(1, counts['total'])
    summary = (
        f'=== Parameter Trainability (stage={stage}) ===\n'
        f"Total: {counts['total']:,} | Trainable: {counts['trainable']:,} ({ratio:.2%})\n"
        f"Backbone: {counts['backbone']:,} | Trainable: {counts['backbone_trainable']:,}\n"
        f"Neck: {counts['neck']:,} | Trainable: {counts['neck_trainable']:,}\n"
        f"ROI Head: {counts['roi_head']:,} | Trainable: {counts['roi_head_trainable']:,}\n"
        f"MACL: {counts['macl']:,} | Trainable: {counts['macl_trainable']:,}\n"
        f"MSP: {counts['msp']:,} | Trainable: {counts['msp_trainable']:,}\n"
        f"DHN: {counts['dhn']:,} | Trainable: {counts['dhn_trainable']:,}\n"
        f"MAN: {counts['man']:,} | Trainable: {counts['man_trainable']:,}\n"
        f"Others: {counts['others']:,} | Trainable: {counts['others_trainable']:,}\n"
    )
    print_and_log(report_path, summary)

    # Stage expectations
    warn = []
    if stage == 'stage2':
        # 期望 backbone 冻结
        if counts['backbone_trainable'] > 0:
            warn.append('[WARN] Stage2 期望 backbone 冻结，但检测到可训练参数 > 0')
    else:
        # Stage1 或 Stage3 期望 backbone 可训练
        if counts['backbone_trainable'] == 0:
            warn.append(f'[WARN] {stage} 期望 backbone 可训练，但检测到 0')

    for w in warn:
        print_and_log(report_path, w)

    # 额外报告：lambda 权重与 BN 冻结统计
    try:
        lambda_info = {}
        if hasattr(model, 'module'):
            _m = model.module
        else:
            _m = model
        roi_head = getattr(_m, 'roi_head', None)
        if roi_head is not None:
            for lk in ['lambda1', 'lambda2', 'lambda3']:
                if hasattr(roi_head, lk):
                    lambda_info[lk] = getattr(roi_head, lk)
        if lambda_info:
            print_and_log(report_path, f"[LAMBDA] {lambda_info}")
    except Exception:
        pass

    try:
        import torch.nn as nn
        total_bn = 0
        frozen_bn = 0
        for mod in _m.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                total_bn += 1
                # 若其参数全部 require_grad=False 视为冻结
                params = list(mod.parameters())
                if params and all(not p.requires_grad for p in params):
                    frozen_bn += 1
        if total_bn > 0:
            print_and_log(report_path, f"[BN] total={total_bn}, frozen={frozen_bn}")
    except Exception:
        pass

    print_and_log(report_path, '参数可训练性检查完成。')


if __name__ == '__main__':
    main()
