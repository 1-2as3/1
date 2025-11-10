"""
Stage1 Weights Solution

问题：Stage1 训练后没有 epoch_latest.pth，但有 29 个 best_epoch_*.pth 文件
解决方案：分析所有 checkpoint，找到最优权重并创建符号链接

使用方法：
    python scripts/solve_stage1_weights.py
    python scripts/solve_stage1_weights.py --create-link  # 自动创建 epoch_latest.pth
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Stage1WeightsSolver:
    """Solve Stage1 missing epoch_latest.pth issue."""
    
    def __init__(self, work_dir: str = 'work_dirs/stage1_llvip_pretrain'):
        self.work_dir = Path(work_dir)
        
        if not self.work_dir.exists():
            raise FileNotFoundError(f"Work directory not found: {work_dir}")
    
    def find_all_checkpoints(self) -> List[Path]:
        """Find all .pth checkpoint files."""
        checkpoints = []
        
        for pattern in ['best_epoch_*.pth', 'epoch_*.pth', 'latest.pth']:
            checkpoints.extend(self.work_dir.glob(pattern))
        
        return sorted(checkpoints)
    
    def parse_checkpoint_info(self, ckpt_path: Path) -> Optional[Dict]:
        """Parse checkpoint metadata from filename.
        
        Format examples:
            - best_epoch_coco_bbox_mAP_epoch_8.pth
            - best_epoch_coco_bbox_mAP_50_epoch_10.pth
            - epoch_12.pth
        
        Returns:
            Dict with keys: file, epoch, metric, value, is_best
        """
        filename = ckpt_path.name
        
        # Pattern 1: best_epoch_<metric>_epoch_<num>.pth
        if filename.startswith('best_epoch_'):
            parts = filename.replace('best_epoch_', '').replace('.pth', '').split('_epoch_')
            
            if len(parts) == 2:
                metric_str, epoch_str = parts
                
                try:
                    epoch = int(epoch_str)
                except ValueError:
                    return None
                
                # Metric name
                metric = metric_str.replace('_', ' ')
                
                return {
                    'file': ckpt_path,
                    'epoch': epoch,
                    'metric': metric,
                    'value': None,  # Value not in filename
                    'is_best': True
                }
        
        # Pattern 2: epoch_<num>.pth
        elif filename.startswith('epoch_'):
            try:
                epoch = int(filename.replace('epoch_', '').replace('.pth', ''))
                return {
                    'file': ckpt_path,
                    'epoch': epoch,
                    'metric': None,
                    'value': None,
                    'is_best': False
                }
            except ValueError:
                return None
        
        # Pattern 3: latest.pth
        elif filename == 'latest.pth':
            return {
                'file': ckpt_path,
                'epoch': None,
                'metric': 'latest',
                'value': None,
                'is_best': False
            }
        
        return None
    
    def analyze_checkpoints(self) -> Dict:
        """Analyze all checkpoints and recommend best one.
        
        Returns:
            Analysis report with:
                - total_checkpoints: int
                - checkpoint_info: List[Dict]
                - best_by_metric: Dict[str, Path]
                - recommended: Path
                - reason: str
        """
        all_checkpoints = self.find_all_checkpoints()
        
        if not all_checkpoints:
            return {
                'total_checkpoints': 0,
                'checkpoint_info': [],
                'best_by_metric': {},
                'recommended': None,
                'reason': 'No checkpoints found'
            }
        
        # Parse all checkpoints
        checkpoint_info = []
        for ckpt in all_checkpoints:
            info = self.parse_checkpoint_info(ckpt)
            if info:
                checkpoint_info.append(info)
        
        # Group by metric
        best_by_metric = {}
        highest_epoch = None
        
        for info in checkpoint_info:
            metric = info.get('metric')
            epoch = info.get('epoch')
            
            if metric and info['is_best']:
                if metric not in best_by_metric:
                    best_by_metric[metric] = info['file']
                else:
                    # Keep the one with higher epoch
                    existing_info = next(
                        i for i in checkpoint_info 
                        if i['file'] == best_by_metric[metric]
                    )
                    if epoch and existing_info.get('epoch'):
                        if epoch > existing_info['epoch']:
                            best_by_metric[metric] = info['file']
            
            # Track highest epoch
            if epoch:
                if highest_epoch is None or epoch > highest_epoch['epoch']:
                    highest_epoch = info
        
        # Recommendation logic
        recommended = None
        reason = ""
        
        # Priority 1: best_epoch_coco_bbox_mAP (primary metric)
        if 'coco bbox mAP' in best_by_metric:
            recommended = best_by_metric['coco bbox mAP']
            reason = "Best checkpoint by coco_bbox_mAP (primary metric)"
        
        # Priority 2: Any best checkpoint
        elif best_by_metric:
            metric, path = next(iter(best_by_metric.items()))
            recommended = path
            reason = f"Best checkpoint by {metric}"
        
        # Priority 3: Highest epoch
        elif highest_epoch:
            recommended = highest_epoch['file']
            reason = f"Highest epoch checkpoint (epoch {highest_epoch['epoch']})"
        
        # Priority 4: First available
        else:
            recommended = all_checkpoints[0]
            reason = "First available checkpoint"
        
        return {
            'total_checkpoints': len(all_checkpoints),
            'checkpoint_info': checkpoint_info,
            'best_by_metric': best_by_metric,
            'recommended': recommended,
            'reason': reason
        }
    
    def create_epoch_latest_link(self, source: Path, dry_run: bool = True) -> bool:
        """Create epoch_latest.pth pointing to best checkpoint.
        
        Args:
            source: Source checkpoint file
            dry_run: If True, only print what would be done
        
        Returns:
            True if successful
        """
        target = self.work_dir / 'epoch_latest.pth'
        
        print(f"\nCreating epoch_latest.pth link:")
        print(f"  Source: {source.name}")
        print(f"  Target: {target}")
        
        if target.exists():
            print(f"  ⚠️  Target already exists")
            
            if not dry_run:
                backup = target.with_suffix('.pth.backup')
                print(f"  Creating backup: {backup.name}")
                shutil.copy2(target, backup)
        
        if dry_run:
            print(f"  [DRY RUN] Would copy {source.name} -> epoch_latest.pth")
            return True
        
        try:
            # Use copy instead of symlink (better compatibility on Windows)
            shutil.copy2(source, target)
            print(f"  ✓ Successfully created epoch_latest.pth")
            return True
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    def print_report(self, analysis: Dict, create_link: bool = False):
        """Print analysis report."""
        print("=" * 70)
        print("Stage1 Weights Analysis")
        print("=" * 70)
        print(f"Work Directory: {self.work_dir}")
        print(f"Total Checkpoints: {analysis['total_checkpoints']}")
        print()
        
        if not analysis['checkpoint_info']:
            print("⚠️  No checkpoints found!")
            return
        
        # List all checkpoints
        print("Available Checkpoints:")
        for info in sorted(analysis['checkpoint_info'], key=lambda x: x.get('epoch', 0)):
            epoch = info.get('epoch', 'N/A')
            metric = info.get('metric', 'regular')
            is_best = ' [BEST]' if info['is_best'] else ''
            print(f"  - {info['file'].name:50s} Epoch: {epoch:3s} {metric}{is_best}")
        print()
        
        # Best by metric
        if analysis['best_by_metric']:
            print("Best Checkpoints by Metric:")
            for metric, path in analysis['best_by_metric'].items():
                print(f"  - {metric:30s} -> {path.name}")
            print()
        
        # Recommendation
        print("=" * 70)
        print("Recommendation:")
        print(f"  File: {analysis['recommended'].name}")
        print(f"  Reason: {analysis['reason']}")
        print("=" * 70)
        
        # Create link option
        if create_link:
            print()
            success = self.create_epoch_latest_link(
                analysis['recommended'], 
                dry_run=False
            )
            
            if success:
                print("\n✓ epoch_latest.pth created successfully!")
                print("  You can now use this in Stage2 config:")
                print("  load_from = 'work_dirs/stage1_llvip_pretrain/epoch_latest.pth'")
        else:
            print("\nTo create epoch_latest.pth, run:")
            print("  python scripts/solve_stage1_weights.py --create-link")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Solve Stage1 missing epoch_latest.pth issue'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dirs/stage1_llvip_pretrain',
        help='Stage1 work directory (default: work_dirs/stage1_llvip_pretrain)'
    )
    parser.add_argument(
        '--create-link',
        action='store_true',
        help='Create epoch_latest.pth symlink to recommended checkpoint'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output analysis as JSON'
    )
    
    args = parser.parse_args()
    
    try:
        solver = Stage1WeightsSolver(args.work_dir)
        analysis = solver.analyze_checkpoints()
        
        if args.json:
            # Convert Path objects to strings for JSON
            json_analysis = {
                'total_checkpoints': analysis['total_checkpoints'],
                'checkpoint_info': [
                    {**info, 'file': str(info['file'])} 
                    for info in analysis['checkpoint_info']
                ],
                'best_by_metric': {
                    k: str(v) for k, v in analysis['best_by_metric'].items()
                },
                'recommended': str(analysis['recommended']) if analysis['recommended'] else None,
                'reason': analysis['reason']
            }
            print(json.dumps(json_analysis, indent=2))
        else:
            solver.print_report(analysis, args.create_link)
    
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
