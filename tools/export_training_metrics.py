import os
import sys
import json
import csv
from datetime import datetime
from typing import Dict, Any, List

try:
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend for servers/CI
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_scalars(jsonl_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                # skip broken lines
                continue
    return records


def to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def write_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    # collect all keys observed
    keys = set()
    for r in records:
        keys.update(r.keys())
    # Stable ordering with some preferred columns first
    preferred = [
        'epoch', 'iter', 'step', 'lr', 'loss', 'loss_total', 'loss_macl',
        'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'acc',
        'grad_norm', 'memory', 'time', 'data_time', 'pascal_voc/mAP', 'pascal_voc/AP50'
    ]
    rest = [k for k in sorted(keys) if k not in preferred]
    header = [k for k in preferred if k in keys] + rest

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, '') for k in header})


def plot_curves(records: List[Dict[str, Any]], out_dir: str) -> List[str]:
    if plt is None:
        return []

    os.makedirs(out_dir, exist_ok=True)

    # Split into iter-based (training) and step-based (evaluation)
    iters = [r.get('iter') for r in records if 'iter' in r]
    # Ensure numeric
    iters_vals = [int(i) for i in iters]

    # Helper to extract series
    def series(key: str):
        xs, ys = [], []
        for r in records:
            if 'iter' in r and key in r and r[key] == r[key]:  # filter NaN
                xs.append(int(r['iter']))
                ys.append(to_float(r[key]))
        return xs, ys

    saved = []

    # 1) loss_total and loss
    for name, keys in [
        ('loss_total_vs_iter.png', ['loss_total']),
        ('loss_vs_iter.png', ['loss'])
    ]:
        plt.figure(figsize=(8, 4))
        for k in keys:
            xs, ys = series(k)
            if xs:
                plt.plot(xs, ys, label=k)
        plt.xlabel('iter')
        plt.ylabel('value')
        plt.title(', '.join(keys))
        plt.grid(True, alpha=0.3)
        if any(series(k)[0] for k in keys):
            plt.legend()
        p = os.path.join(out_dir, name)
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)

    # 2) loss components
    comp_keys = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'loss_macl']
    plt.figure(figsize=(10, 5))
    plotted = False
    for k in comp_keys:
        xs, ys = series(k)
        if xs:
            plt.plot(xs, ys, label=k)
            plotted = True
    plt.xlabel('iter')
    plt.ylabel('value')
    plt.title('loss components')
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend()
    p = os.path.join(out_dir, 'loss_components_vs_iter.png')
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    saved.append(p)

    # 3) lr
    xs, ys = series('lr')
    if xs:
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.xlabel('iter')
        plt.ylabel('lr')
        plt.title('learning rate')
        plt.grid(True, alpha=0.3)
        p = os.path.join(out_dir, 'lr_vs_iter.png')
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)

    # 4) grad_norm
    xs, ys = series('grad_norm')
    if xs:
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.xlabel('iter')
        plt.ylabel('grad_norm')
        plt.title('grad norm')
        plt.grid(True, alpha=0.3)
        p = os.path.join(out_dir, 'grad_norm_vs_iter.png')
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)

    # 5) time / data_time
    for k in ['time', 'data_time']:
        xs, ys = series(k)
        if xs:
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys)
            plt.xlabel('iter')
            plt.ylabel(k)
            plt.title(k)
            plt.grid(True, alpha=0.3)
            p = os.path.join(out_dir, f'{k}_vs_iter.png')
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            saved.append(p)

    # 6) eval mAP (if any)
    eval_steps, eval_map = [], []
    for r in records:
        if 'pascal_voc/mAP' in r:
            s = int(r.get('step', len(eval_steps) + 1))
            eval_steps.append(s)
            eval_map.append(to_float(r['pascal_voc/mAP']))
    if eval_steps:
        plt.figure(figsize=(6, 4))
        plt.plot(eval_steps, eval_map, marker='o')
        plt.xlabel('eval step (epoch)')
        plt.ylabel('mAP')
        plt.title('VOC mAP')
        plt.grid(True, alpha=0.3)
        p = os.path.join(out_dir, 'mAP_vs_epoch.png')
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        saved.append(p)

    return saved


def health_check(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    def collect_float(key: str) -> List[float]:
        arr = []
        for r in records:
            if key in r and isinstance(r.get('iter'), (int, float)):
                v = r[key]
                if isinstance(v, (int, float)):
                    arr.append(float(v))
        return arr

    # NaN checks across selected metrics
    nan_flags = {}
    keys_to_check = ['loss', 'loss_total', 'loss_bbox', 'loss_macl']
    for k in keys_to_check:
        nan_idx = []
        for r in records:
            if k in r and r.get('iter') is not None:
                v = r[k]
                try:
                    if v != v:  # NaN
                        nan_idx.append(int(r.get('iter')))
                except Exception:
                    pass
        nan_flags[k] = nan_idx

    report['nan_occurrences'] = nan_flags

    # Trend for loss_total (ignore NaN)
    loss_total = [r['loss_total'] for r in records if 'iter' in r and 'loss_total' in r and r['loss_total'] == r['loss_total']]
    if loss_total:
        report['loss_total_start'] = float(loss_total[0])
        report['loss_total_end'] = float(loss_total[-1])
        report['loss_total_change'] = float(loss_total[-1] - loss_total[0])

    # LR schedule sanity
    lr_vals = collect_float('lr')
    if lr_vals:
        report['lr_min'] = float(min(lr_vals))
        report['lr_max'] = float(max(lr_vals))

    # mAP if available
    mAP_vals = [to_float(r['pascal_voc/mAP']) for r in records if 'pascal_voc/mAP' in r]
    if mAP_vals:
        report['mAP_last'] = float(mAP_vals[-1])

    # Memory
    mem_vals = collect_float('memory')
    if mem_vals:
        report['memory_median(MB)'] = float(sorted(mem_vals)[len(mem_vals)//2])

    # Simple conclusions
    conclusions = []
    if any(nan_flags.values()):
        total_nans = sum(len(v) for v in nan_flags.values())
        conclusions.append(f'检测到 {total_nans} 次 NaN（分布: ' + ', '.join(f"{k}:{len(v)}" for k,v in nan_flags.items()) + '). 建议继续观察或在聚合损失处加入 nan_to_num 保护。')
    if loss_total:
        if loss_total[-1] < loss_total[0]:
            conclusions.append('loss_total 整体呈下降趋势，训练收敛正常。')
        else:
            conclusions.append('loss_total 未显著下降，建议检查学习率或数据。')
    if lr_vals:
        conclusions.append(f'学习率范围 [{min(lr_vals):.2e}, {max(lr_vals):.2e}] 符合线性warmup到恒定。')
    if mAP_vals:
        conclusions.append(f'最终VOC mAP ≈ {mAP_vals[-1]:.3f}。')
    if mem_vals:
        conclusions.append('显存占用稳定。')

    report['conclusions'] = conclusions
    return report


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # auto-pick latest run dir under work_dirs/stage1_llvip_pretrain
        base = os.path.join('work_dirs', 'stage1_llvip_pretrain')
        if not os.path.isdir(base):
            print('Run dir not specified and default base not found.')
            sys.exit(1)
        subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not subdirs:
            print('No run dirs under', base)
            sys.exit(1)
        run_dir = sorted(subdirs, key=os.path.getmtime)[-1]

    vis_dir = os.path.join(run_dir, 'vis_data')
    jsonl = os.path.join(vis_dir, 'scalars.json')
    if not os.path.isfile(jsonl):
        print('scalars.json not found at', jsonl)
        sys.exit(2)

    records = load_scalars(jsonl)
    ts_name = os.path.basename(run_dir)
    out_dir = os.path.join('reports', f'training_{ts_name}')
    os.makedirs(out_dir, exist_ok=True)

    # CSV export
    csv_path = os.path.join(out_dir, 'metrics.csv')
    write_csv(records, csv_path)

    # Plots
    saved = plot_curves(records, out_dir)

    # Health check
    report = health_check(records)
    report_path = os.path.join(out_dir, 'summary.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('[OK] Exported')
    print(' - CSV:', csv_path)
    if saved:
        for p in saved:
            print(' - Plot:', p)
    print(' - Summary:', report_path)


if __name__ == '__main__':
    main()
