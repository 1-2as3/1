#!/usr/bin/env python
"""Parse training log and extract mAP/recall/loss trends.

Usage:
    python tools/parse_training_log.py work_dirs/stage2_1_kaist_detonly/xxx/xxx.log
"""
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_log(log_file):
    """Extract training metrics from MMDetection log."""
    metrics = defaultdict(list)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract epoch validation mAP
            # Example: "Epoch(val) [2][4168/4168]    pascal_voc/mAP: 0.5908"
            match = re.search(r'Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*pascal_voc/mAP:\s*([\d\.]+)', line)
            if match:
                epoch = int(match.group(1))
                map_val = float(match.group(2))
                metrics['epoch'].append(epoch)
                metrics['mAP'].append(map_val)
            
            # Extract AP50 if exists
            match_ap50 = re.search(r'pascal_voc/AP50:\s*([\d\.]+)', line)
            if match_ap50:
                metrics['AP50'].append(float(match_ap50.group(1)))
            
            # Extract recall from VOC table
            # Example: "| person | 6870 | 14301 | 0.734  | 0.591 |"
            match_recall = re.search(r'\|\s*person\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d\.]+)\s*\|', line)
            if match_recall:
                metrics['recall'].append(float(match_recall.group(1)))
            
            # Extract training loss
            # Example: "Epoch(train) [2][  50/11439]  ... loss_total: 0.1882"
            match_loss = re.search(r'Epoch\(train\) \[(\d+)\]\[.*?\].*loss_total:\s*([\d\.]+)', line)
            if match_loss:
                epoch = int(match_loss.group(1))
                loss = float(match_loss.group(2))
                if 'train_epoch' not in metrics or metrics['train_epoch'][-1] != epoch:
                    metrics['train_epoch'].append(epoch)
                    metrics['avg_loss'].append(loss)
    
    return metrics


def print_summary(metrics):
    """Print formatted summary."""
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    
    if metrics['mAP']:
        print(f"\n🎯 Validation mAP Trend:")
        for i, (epoch, map_val) in enumerate(zip(metrics['epoch'], metrics['mAP'])):
            recall = metrics['recall'][i] if i < len(metrics['recall']) else 'N/A'
            arrow = ""
            if i > 0:
                diff = map_val - metrics['mAP'][i-1]
                arrow = f"{'📈' if diff > 0 else '📉'} {diff:+.4f}"
            print(f"  Epoch {epoch:2d}: mAP = {map_val:.4f}  Recall = {recall}  {arrow}")
        
        best_epoch = metrics['epoch'][metrics['mAP'].index(max(metrics['mAP']))]
        best_map = max(metrics['mAP'])
        print(f"\n✨ Best: Epoch {best_epoch}, mAP = {best_map:.4f}")
        
        # Trend analysis
        if len(metrics['mAP']) >= 2:
            recent_trend = metrics['mAP'][-1] - metrics['mAP'][-2]
            status = "✅ Improving" if recent_trend > 0 else "⚠️ Declining"
            print(f"\n📈 Recent Trend: {status} ({recent_trend:+.4f})")
            
            # Early warning
            if metrics['mAP'][-1] < 0.60:
                print(f"\n🚨 WARNING: Latest mAP ({metrics['mAP'][-1]:.4f}) < 0.60!")
                print("   Consider stopping and adjusting hyperparameters.")
    
    if metrics['avg_loss']:
        print(f"\n🔧 Training Loss (per epoch average):")
        for epoch, loss in zip(metrics['train_epoch'], metrics['avg_loss']):
            print(f"  Epoch {epoch:2d}: {loss:.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Parse MMDetection training log')
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--plot', action='store_true', help='Generate plot (requires matplotlib)')
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"📖 Parsing log: {log_path}")
    metrics = parse_log(log_path)
    
    if not metrics['mAP']:
        print("⚠️ No validation metrics found in log file.")
        sys.exit(1)
    
    print_summary(metrics)
    
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot mAP
            ax1.plot(metrics['epoch'], metrics['mAP'], 'o-', linewidth=2, markersize=8)
            ax1.axhline(y=0.60, color='orange', linestyle='--', label='Warning Threshold (0.60)')
            ax1.axhline(y=0.63, color='green', linestyle='--', label='Target (0.63)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('mAP')
            ax1.set_title('Validation mAP Trend')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot recall
            if metrics['recall']:
                ax2.plot(metrics['epoch'], metrics['recall'], 's-', color='purple', linewidth=2, markersize=8)
                ax2.axhline(y=0.80, color='green', linestyle='--', label='Target Recall (0.80)')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Recall')
                ax2.set_title('Validation Recall Trend')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = log_path.parent / f"{log_path.stem}_metrics.png"
            plt.savefig(output_path, dpi=150)
            print(f"\n💾 Plot saved to: {output_path}")
            
        except ImportError:
            print("\n⚠️ matplotlib not installed. Skipping plot generation.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
