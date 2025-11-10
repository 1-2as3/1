import re
import os
import csv
import argparse
import math
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

LOG_PATTERN = re.compile(r"\[TSNEVisualHook\] Alignment metrics - Inter-modal dist: ([0-9\.]+), Intra-modal dist \(vis/ir\): ([0-9\.]+)/([0-9\.]+), Alignment score: ([0-9\.]+)")
EPOCH_PATTERN = re.compile(r"Epoch\(train\)\s+\[(\d+)\]")
LOSS_PATTERN = re.compile(r"Epoch\(train\)\s+\[(\d+)\].*?grad_norm: ([0-9\.]+).*?loss_macl: ([0-9\.]+).*?loss_total: ([0-9\.]+)")

def parse_log(log_path: str):
    align_rows = []
    loss_rows = {}
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '[TSNEVisualHook] Alignment metrics' in line:
                m = LOG_PATTERN.search(line)
                # 从上一行 epoch 信息中提取 epoch（日志里通常紧邻）
                # 保险起见重新搜索 epoch 索引
                ep_search = re.search(r"Epoch\(train\)\s+\[(\d+)\]", line)
                epoch = int(ep_search.group(1)) if ep_search else None
                if m and epoch is not None:
                    align_rows.append(dict(epoch=epoch,
                                           inter=float(m.group(1)),
                                           vis_intra=float(m.group(2)),
                                           ir_intra=float(m.group(3)),
                                           score=float(m.group(4))))
            elif 'Epoch(train)' in line and 'loss_macl' in line:
                lm = LOSS_PATTERN.search(line)
                if lm:
                    ep = int(lm.group(1))
                    loss_rows[ep] = dict(
                        grad_norm=float(lm.group(2)),
                        loss_macl=float(lm.group(3)),
                        loss_total=float(lm.group(4))
                    )
    # 合并
    merged = []
    for row in align_rows:
        ep = row['epoch']
        losses = loss_rows.get(ep, {})
        merged.append({**row, **losses})
    return merged

def summarize(results):
    if not results:
        return {}
    scores = [r['score'] for r in results]
    losses = [r.get('loss_total', math.nan) for r in results if 'loss_total' in r]
    macls = [r.get('loss_macl', math.nan) for r in results if 'loss_macl' in r]
    return {
        'epochs': len(results),
        'score_mean': sum(scores)/len(scores),
        'score_min': min(scores),
        'score_max': max(scores),
        'score_std': (sum((s - (sum(scores)/len(scores)))**2 for s in scores)/len(scores))**0.5,
        'loss_total_last': losses[-1] if losses else None,
        'loss_macl_last': macls[-1] if macls else None,
        'first_score': scores[0],
        'last_score': scores[-1]
    }

def write_csv(results, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ['epoch','inter','vis_intra','ir_intra','score','grad_norm','loss_macl','loss_total']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

def plot_curves(results, out_dir):
    if not _HAS_PLT or not results:
        return
    os.makedirs(out_dir, exist_ok=True)
    epochs = [r['epoch'] for r in results]
    score = [r['score'] for r in results]
    loss_total = [r.get('loss_total') for r in results]
    loss_macl = [r.get('loss_macl') for r in results]
    grad_norm = [r.get('grad_norm') for r in results]

    plt.figure(figsize=(10,4))
    plt.plot(epochs, score, marker='o', label='Alignment Score')
    plt.title('Alignment Score Trend')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'alignment_score.png'))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, loss_total, marker='o', label='loss_total')
    plt.plot(epochs, loss_macl, marker='x', label='loss_macl')
    plt.title('Loss Trends')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'loss_trends.png'))
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(epochs, grad_norm, marker='s', label='grad_norm')
    plt.title('Grad Norm Trend')
    plt.xlabel('Epoch'); plt.ylabel('grad_norm'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'grad_norm.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Summarize TSNE alignment metrics from log.')
    parser.add_argument('--log', required=True, help='Path to training log file')
    parser.add_argument('--out-csv', default='work_dirs/alignment_summary/alignment_metrics.csv')
    parser.add_argument('--plot-dir', default='work_dirs/alignment_summary')
    args = parser.parse_args()

    res = parse_log(args.log)
    if not res:
        print('[summarize_alignment] No alignment metrics found.')
        return
    write_csv(res, args.out_csv)
    stat = summarize(res)
    print('[summarize_alignment] Summary:')
    for k,v in stat.items():
        print(f'  {k}: {v}')
    plot_curves(res, args.plot_dir)

if __name__ == '__main__':
    main()
