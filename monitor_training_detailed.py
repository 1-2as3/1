#!/usr/bin/env python3
"""
Real-time Training Monitor for Stage2.1 Recovery
Analyzes log files and visualizes training progress
"""
import os
import re
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """Parse training log and extract metrics"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    train_metrics = []
    val_metrics = []
    
    for line in lines:
        # Parse training progress
        train_match = re.search(
            r'Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\s*(\d+)\].*'
            r'loss_total:\s*([\d.]+)',
            line
        )
        if train_match:
            # Extract additional metrics
            loss_cls_match = re.search(r'loss_cls:\s*([\d.]+)', line)
            loss_bbox_match = re.search(r'loss_bbox:\s*([\d.]+)', line)
            acc_match = re.search(r'acc:\s*([\d.]+)', line)
            
            epoch, batch, total, loss_total = train_match.groups()
            train_metrics.append({
                'epoch': int(epoch),
                'batch': int(batch),
                'total_batches': int(total),
                'loss_total': float(loss_total),
                'loss_cls': float(loss_cls_match.group(1)) if loss_cls_match else 0.0,
                'loss_bbox': float(loss_bbox_match.group(1)) if loss_bbox_match else 0.0,
                'acc': float(acc_match.group(1)) if acc_match else 0.0
            })
            continue
        
        # Legacy format (for older logs)
        train_match_legacy = re.search(
            r'Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\s*(\d+)\].*'
            r'loss_cls:\s*([\d.]+).*'
            r'loss_bbox:\s*([\d.]+).*'
            r'acc:\s*([\d.]+)',
            line
        )
        if train_match_legacy:
            epoch, batch, total, loss_cls, loss_bbox, acc = train_match_legacy.groups()
            # Try to find loss_total in the line
            loss_total_match = re.search(r'loss_total:\s*([\d.]+)', line)
            train_metrics.append({
                'epoch': int(epoch),
                'batch': int(batch),
                'total_batches': int(total),
                'loss_total': float(loss_total_match.group(1)) if loss_total_match else 0.0,
                'loss_cls': float(loss_cls),
                'loss_bbox': float(loss_bbox),
                'acc': float(acc)
            })
            continue
        
        # Parse validation results
        val_match = re.search(r'pascal_voc/mAP:\s*([\d.]+)', line)
        if val_match:
            # Extract epoch from previous lines
            epoch_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
            else:
                # Try to infer from training metrics
                epoch = train_metrics[-1]['epoch'] if train_metrics else 0
            
            val_metrics.append({
                'epoch': epoch,
                'mAP': float(val_match.group(1))
            })
    
    return train_metrics, val_metrics


def get_latest_log(work_dir):
    """Find the most recent log file"""
    work_path = Path(work_dir)
    if not work_path.exists():
        return None
    
    log_files = list(work_path.glob('*/*.log'))
    if not log_files:
        return None
    
    # Sort by modification time
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    return latest_log


def print_summary(train_metrics, val_metrics, log_path):
    """Print training summary"""
    print("=" * 80)
    print(f"Plan A - Pure Detection Training Monitor")
    print("=" * 80)
    print(f"Log file: {log_path}")
    print(f"Last updated: {datetime.fromtimestamp(log_path.stat().st_mtime)}")
    print()
    
    if train_metrics:
        latest = train_metrics[-1]
        progress = (latest['batch'] / latest['total_batches']) * 100
        
        print(f"[CURRENT PROGRESS]")
        print(f"  Epoch: {latest['epoch']}")
        print(f"  Batch: {latest['batch']}/{latest['total_batches']} ({progress:.1f}%)")
        print(f"  Loss Total: {latest['loss_total']:.4f}")
        print(f"  Loss Cls: {latest['loss_cls']:.4f}")
        print(f"  Loss BBox: {latest['loss_bbox']:.4f}")
        print(f"  Accuracy: {latest['acc']:.2f}%")
        print()
        
        # Calculate loss trends (last 100 batches)
        recent = train_metrics[-100:] if len(train_metrics) > 100 else train_metrics
        avg_loss = sum(m['loss_total'] for m in recent) / len(recent)
        print(f"[LOSS TREND - Last {len(recent)} batches]")
        print(f"  Average loss_total: {avg_loss:.4f}")
        if len(recent) >= 2:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            first_avg = sum(m['loss_total'] for m in first_half) / len(first_half)
            second_avg = sum(m['loss_total'] for m in second_half) / len(second_half)
            trend = "↓ Decreasing" if second_avg < first_avg else "↑ Increasing"
            print(f"  Trend: {trend} ({first_avg:.4f} → {second_avg:.4f})")
        print()
    
    if val_metrics:
        print(f"[VALIDATION RESULTS]")
        print(f"  Completed epochs: {len(val_metrics)}")
        print()
        for vm in val_metrics:
            status = ""
            if vm['mAP'] >= 0.63:
                status = "✓ TARGET REACHED!"
            elif vm['mAP'] >= 0.60:
                status = "↑ Good progress"
            elif vm['mAP'] >= 0.5265:
                status = "→ Above failed baseline"
            else:
                status = "⚠ Below failed baseline"
            print(f"  Epoch {vm['epoch']}: mAP = {vm['mAP']:.4f}  {status}")
        print()
        
        if len(val_metrics) >= 2:
            latest_mAP = val_metrics[-1]['mAP']
            prev_mAP = val_metrics[-2]['mAP']
            delta = latest_mAP - prev_mAP
            print(f"  Latest change: {delta:+.4f}")
        print()
    
    print("=" * 80)
    print("[BASELINES]")
    print("  Stage1 epoch_21:     mAP = 0.6288  (success)")
    print("  Stage2.1 epoch_3:    mAP = 0.5265  (failed)")
    print("  Recovery target:     mAP ≥ 0.63")
    print("=" * 80)


def main():
    work_dir = 'work_dirs/stage2_1_pure_detection'
    
    print("Searching for latest log file...")
    log_path = get_latest_log(work_dir)
    
    if not log_path:
        print(f"ERROR: No log files found in {work_dir}")
        return
    
    print(f"Found: {log_path}\n")
    
    train_metrics, val_metrics = parse_log_file(log_path)
    print_summary(train_metrics, val_metrics, log_path)


if __name__ == '__main__':
    main()
