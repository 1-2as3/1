"""
Research Preflight Runner

一键执行：依赖检查、GPU/显存、参数可训练性（各阶段）、Loss 分项、梯度流占位、数据集加载占位、
并输出最终 PASS/FAIL 与修复建议。

特点：
- 对缺失依赖/数据/日志保持容错，不中断整体流程，输出明确的建议与下一步动作。
- Windows 友好，使用 sys.executable 调用子脚本。
- 汇总到 logs/research_validation_report.txt 与 logs/research_preflight_summary.json。
"""
import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_TXT = Path('logs/research_validation_report.txt')
REPORT_JSON = Path('logs/research_preflight_summary.json')


def log(line: str):
    print(line)
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def safe_import(module: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(module)
        return True, None
    except Exception as e:
        return False, str(e)


def run_script(script: Path, args: List[str]) -> Dict[str, Any]:
    cmd = [sys.executable, str(script)] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        return {
            'ok': proc.returncode == 0,
            'code': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'cmd': cmd,
        }
    except Exception as e:
        return {'ok': False, 'code': -1, 'stdout': '', 'stderr': str(e), 'cmd': cmd}


def gpu_memory_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {'available': False, 'devices': []}
    ok, err = safe_import('torch')
    if not ok:
        log(f"[GPU] PyTorch 未安装，跳过显存检查：{err}")
        return info

    import torch
    if not torch.cuda.is_available():
        log('[GPU] CUDA 不可用，跳过显存检查。')
        return info

    info['available'] = True
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            info['devices'].append({
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'free_bytes': int(free),
                'total_bytes': int(total)
            })
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            log(f"[GPU] GPU{i} {torch.cuda.get_device_name(i)}: Free {free_gb:.2f} GB / Total {total_gb:.2f} GB")
        except Exception as e:
            log(f"[GPU] 获取 GPU{i} 显存失败：{e}")

    # 建议 batch size（粗略启发式）
    if info['devices']:
        max_free = max(d['free_bytes'] for d in info['devices']) / (1024**3)
        if max_free < 8:
            log('[GPU] ⚠️ 可用显存 < 8GB，建议减小 batch size 或使用混合精度。')
        elif max_free < 12:
            log('[GPU] 建议 batch size ≈ 2~4；若模型较大，考虑 2。')
        elif max_free < 24:
            log('[GPU] 建议 batch size ≈ 4~8；可从 4 起步。')
        else:
            log('[GPU] 建议 batch size ≥ 8；可根据 loss 稳定性调整。')

    return info


def stage_param_check(cfg_path: Path, checkpoint: Optional[Path], stage: str) -> Dict[str, Any]:
    """调用 tools/param_check.py，统计可训练参数并进行阶段期望提示。"""
    script = Path('tools/param_check.py')
    if not script.exists():
        log('[PARAM] 缺少 tools/param_check.py，已跳过。')
        return {'ok': False, 'reason': 'missing script'}

    args = [str(cfg_path)]
    if checkpoint:
        args += ['--checkpoint', str(checkpoint)]
    args += ['--stage', stage]

    res = run_script(script, args)
    if res['ok']:
        log(f"[PARAM] {cfg_path.name} 检查完成。")
    else:
        log(f"[PARAM] {cfg_path.name} 检查失败，code={res['code']}\nSTDERR: {res['stderr']}")
    return res


def loss_detail_check(log_file: Optional[Path], stage: str) -> Dict[str, Any]:
    """调用 tools/loss_detail_check.py 对训练日志做分项检查。"""
    script = Path('tools/loss_detail_check.py')
    if not script.exists():
        log('[LOSS] 缺少 tools/loss_detail_check.py，已跳过。')
        return {'ok': False, 'reason': 'missing script'}
    if not log_file or not log_file.exists():
        log('[LOSS] 未提供有效的训练日志路径，已跳过。')
        return {'ok': False, 'reason': 'no log'}

    res = run_script(script, [str(log_file), '--stage', stage, '--json'])
    if res['ok']:
        log('[LOSS] 分项检查通过。')
    else:
        log(f"[LOSS] 分项检查失败，code={res['code']}\nSTDERR: {res['stderr']}")
    return res


def grad_flow_placeholder(cfg_path: Path, checkpoint: Optional[Path], stage: str) -> Dict[str, Any]:
    """调用 tools/grad_flow_check.py 生成梯度图（若无法运行 batch 则做占位提示）。"""
    script = Path('tools/grad_flow_check.py')
    if not script.exists():
        log('[GRAD] 缺少 tools/grad_flow_check.py，已跳过。')
        return {'ok': False, 'reason': 'missing script'}

    args = [str(cfg_path), '--stage', stage]
    if checkpoint:
        args += ['--checkpoint', str(checkpoint)]

    res = run_script(script, args)
    if res['ok']:
        log('[GRAD] 梯度流检查完成（如未提供 batch 则为占位）。')
    else:
        log(f"[GRAD] 梯度流检查失败，code={res['code']}\nSTDERR: {res['stderr']}")
    return res


def dataset_load_placeholder(name: str, root: Optional[Path]):
    if not root:
        log(f"[DATA] {name}: 未提供数据路径，跳过加载验证。")
        return {'ok': False, 'reason': 'no path'}
    ok, err = safe_import('torch')
    ok2, err2 = safe_import('mmdet')
    if not (ok and ok2):
        log(f"[DATA] {name}: 依赖缺失（torch/mmdet），跳过加载验证。")
        return {'ok': False, 'reason': 'missing deps'}
    # 这里保留占位：真实实现需结合项目自定义的 Dataset 配置与 pipeline
    log(f"[DATA] {name}: 已检测到路径 {root}。如需真实 batch 前向，请提供相应 config/dataloader 设置。")
    return {'ok': True, 'note': 'placeholder'}


def main():
    parser = argparse.ArgumentParser(description='一键科研预检')
    parser.add_argument('--stage1-cfg', type=str, default='configs/llvip/stage1_llvip_pretrain.py')
    parser.add_argument('--stage2-cfg', type=str, default='configs/llvip/stage2_kaist_domain_ft.py')
    parser.add_argument('--stage3-cfg', type=str, default='configs/llvip/stage3_joint_multimodal.py')
    parser.add_argument('--stage1-ckpt', type=str, default='work_dirs/stage1_llvip_pretrain/epoch_latest.pth')
    parser.add_argument('--stage2-log', type=str, default='')
    parser.add_argument('--stage3-log', type=str, default='')
    parser.add_argument('--llvip-root', type=str, default='')
    parser.add_argument('--kaist-root', type=str, default='')
    parser.add_argument('--m3fd-root', type=str, default='')
    args = parser.parse_args()

    # 清理旧报告
    if REPORT_TXT.exists():
        try:
            REPORT_TXT.unlink()
        except Exception:
            pass

    summary: Dict[str, Any] = {
        'env': {},
        'gpu': {},
        'stages': {
            'stage1': {},
            'stage2': {},
            'stage3': {},
        },
        'datasets': {},
        'pass': True,
        'defects': [],
        'suggestions': []
    }

    log('=== 预检查开始 ===')

    # 依赖快速检查
    for mod in ['torch', 'mmcv', 'mmdet', 'mmengine']:
        ok, err = safe_import(mod)
        if ok:
            log(f"[ENV] 依赖可用：{mod}")
        else:
            log(f"[ENV] 依赖缺失：{mod} -> {err}")
            summary['defects'].append(f'missing_dep:{mod}')
            summary['pass'] = False
    summary['env']['python'] = sys.version

    # GPU 显存
    log('\n--- GPU / 显存 ---')
    summary['gpu'] = gpu_memory_report()

    # 阶段参数检查 + 梯度占位 + loss 分项（如有日志）
    stages = [
        ('stage1', Path(args.stage1_cfg), Path(args.stage1_ckpt) if args.stage1_ckpt else None, Path(args.stage2_log) if args.stage2_log else None),
        ('stage2', Path(args.stage2_cfg), None, Path(args.stage2_log) if args.stage2_log else None),
        ('stage3', Path(args.stage3_cfg), None, Path(args.stage3_log) if args.stage3_log else None),
    ]

    for stage_name, cfg_path, ckpt_path, log_path in stages:
        log(f"\n--- {stage_name.upper()} ---")
        stage_res: Dict[str, Any] = {}
        if not cfg_path.exists():
            msg = f'缺少配置文件：{cfg_path}'
            log(f'[STAGE] {msg}')
            summary['defects'].append(f'missing_cfg:{stage_name}')
            summary['pass'] = False
            summary['stages'][stage_name] = {'ok': False, 'reason': 'missing cfg'}
            continue

        # 参数可训练性
        pr = stage_param_check(cfg_path, ckpt_path, stage_name)
        stage_res['param_check'] = pr
        if not pr.get('ok', False):
            summary['defects'].append(f'param_check_fail:{stage_name}')
            summary['pass'] = False
            # 自动尝试生成 fixed config（补齐缺失模型字段）
            try:
                from mmengine.config import Config
                base_cfg = Config.fromfile(str(cfg_path))
                model_dict = base_cfg.get('model', {})
                required = {'backbone', 'rpn_head', 'train_cfg', 'test_cfg'}
                missing = [k for k in required if k not in model_dict]
                if missing:
                    # 加载通用 Faster R-CNN base 模型并合并
                    base_model_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')
                    base_model = base_model_cfg.get('model', {})
                    def _deep_update(b, o):
                        r = dict(b)
                        for kk, vv in o.items():
                            if isinstance(vv, dict) and isinstance(r.get(kk), dict):
                                r[kk] = _deep_update(r[kk], vv)
                            else:
                                r[kk] = vv
                        return r
                    merged = _deep_update(base_model, model_dict)
                    fixed_dir = Path('configs/_auto_fixed')
                    fixed_dir.mkdir(parents=True, exist_ok=True)
                    fixed_path = fixed_dir / f'{cfg_path.stem}.fixed.py'
                    with open(fixed_path, 'w', encoding='utf-8') as f:
                        f.write('# Auto-generated fixed config (merged base model)\n')
                        f.write(f'# Missing keys resolved: {missing}\n')
                        f.write('from mmengine.config import read_base\n\n')
                        # 写入原 base 继承（若存在）
                        bases = base_cfg.get('_base_', [])
                        if bases:
                            f.write('with read_base():\n')
                            for b in bases:
                                f.write(f'    # base: {b}\n')
                            f.write('\n')
                        f.write('model = ' + repr(merged) + '\n')
                        # 复制常见字段
                        for key in ['load_from', 'optim_wrapper', 'param_scheduler', 'work_dir', 'train_dataloader', 'val_dataloader', 'test_dataloader']:
                            if key in base_cfg:
                                f.write(f'\n{key} = {repr(base_cfg[key])}\n')
                    log(f'[AUTO_FIX] 生成修复配置: {fixed_path}')
                    summary['suggestions'].append(f'使用修复版配置: {fixed_path}')
            except Exception as e:
                log(f'[AUTO_FIX] 生成修复配置失败：{e}')

        # 梯度占位
        gr = grad_flow_placeholder(cfg_path, ckpt_path, stage_name)
        stage_res['grad_flow'] = gr

        # loss 分项（如有日志）
        if log_path:
            lr = loss_detail_check(log_path, stage_name)
            stage_res['loss_report'] = lr
            if not lr.get('ok', False):
                summary['defects'].append(f'loss_check_fail:{stage_name}')
                summary['pass'] = False
        else:
            stage_res['loss_report'] = {'ok': False, 'reason': 'no log provided'}

        summary['stages'][stage_name] = stage_res

    # 数据集占位校验
    log('\n--- 数据集加载（占位） ---')
    ds_args = {
        'LLVIP': Path(args.llvip_root) if args.llvip_root else None,
        'KAIST': Path(args.kaist_root) if args.kaist_root else None,
        'M3FD': Path(args.m3fd_root) if args.m3fd_root else None,
    }
    for name, root in ds_args.items():
        dr = dataset_load_placeholder(name, root)
        summary['datasets'][name] = dr
        if not dr.get('ok', False):
            summary['defects'].append(f'dataset_check:{name}')

    # 生成建议
    log('\n--- 建议与后续 ---')
    if any('missing_dep' in d for d in summary['defects']):
        summary['suggestions'].append('安装依赖：pip install -r requirements/runtime.txt')
        log('[SUGGEST] 安装依赖：pip install -r requirements/runtime.txt')
    if any('param_check_fail' in d for d in summary['defects']):
        log('[SUGGEST] 检查阶段期望的冻结策略与 loss 开关，必要时调整 config 并另存到 configs/_auto_fixed/')
        summary['suggestions'].append('调整冻结层/模块开关，另存 fixed 配置')
    if any('loss_check_fail' in d for d in summary['defects']):
        log('[SUGGEST] 检查训练日志路径与日志格式（.log.json），确保自定义 loss 正确写入')
        summary['suggestions'].append('修正日志路径或启用 JSONLogger')

    # 总结
    log('\n=== 预检查总结 ===')
    if summary['pass']:
        log('✅ 所有模块接入与训练逻辑检查基本通过（部分项目可能为占位检测）。')
    else:
        log('✗ 预检查存在问题，请根据上文建议修复后再启动正式训练。')

    # 写入 JSON 汇总
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 退出码：仅当关键步骤均通过时返回 0
    sys.exit(0 if summary['pass'] else 1)


if __name__ == '__main__':
    main()
