"""
Training Loss Verification Script

检查训练日志中是否正确输出 loss_macl, loss_msp, loss_dhn 等自定义 Loss。
验证 MACL/MSP/DHN 模块是否真正激活并贡献到总 loss。
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class LossVerifier:
    """Verify custom losses in training logs."""
    
    # Expected loss patterns for different stages
    EXPECTED_LOSSES = {
        'stage1': ['loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox'],
        'stage2': ['loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox', 
                   'loss_macl', 'loss_msp', 'loss_dhn'],
        'stage3': ['loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox',
                   'loss_macl', 'loss_msp', 'loss_dhn', 'loss_domain_align']
    }
    
    CUSTOM_LOSSES = ['loss_macl', 'loss_msp', 'loss_dhn', 'loss_domain_align']
    
    def __init__(self, log_file: str, stage: str = 'auto'):
        """
        Args:
            log_file: Path to training log file (*.log or *.log.json)
            stage: Which stage to verify ('stage1', 'stage2', 'stage3', or 'auto')
        """
        self.log_file = Path(log_file)
        
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        # Auto-detect stage from filename or directory
        if stage == 'auto':
            if 'stage1' in str(log_file).lower():
                self.stage = 'stage1'
            elif 'stage2' in str(log_file).lower():
                self.stage = 'stage2'
            elif 'stage3' in str(log_file).lower():
                self.stage = 'stage3'
            else:
                self.stage = 'stage2'  # Default to stage2 (most critical)
        else:
            self.stage = stage
        
        self.expected_losses = self.EXPECTED_LOSSES[self.stage]
        self.loss_history = defaultdict(list)
        
    def parse_log_file(self) -> List[Dict]:
        """Parse log file and extract loss entries.
        
        Returns:
            List of loss dictionaries, each containing:
                - epoch: int
                - iter: int
                - loss: float
                - loss_*: float (individual loss values)
        """
        entries = []
        
        if self.log_file.suffix == '.json':
            entries = self._parse_json_log()
        else:
            entries = self._parse_text_log()
        
        # Organize loss history
        for entry in entries:
            for key, value in entry.items():
                if key.startswith('loss'):
                    self.loss_history[key].append(value)
        
        return entries
    
    def _parse_json_log(self) -> List[Dict]:
        """Parse MMEngine JSON log format."""
        entries = []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Only process training mode entries
                    if data.get('mode') != 'train':
                        continue
                    
                    # Extract loss values
                    entry = {}
                    for key, value in data.items():
                        if key in ['epoch', 'iter'] or key.startswith('loss'):
                            entry[key] = value
                    
                    if 'loss' in entry:  # Valid loss entry
                        entries.append(entry)
                
                except json.JSONDecodeError:
                    continue
        
        return entries
    
    def _parse_text_log(self) -> List[Dict]:
        """Parse plain text log format."""
        entries = []
        
        # Pattern: Epoch [1][10/100]  lr: 0.001, loss: 1.234, loss_cls: 0.5, ...
        pattern = re.compile(
            r'Epoch.*?\[(\d+)\]\[(\d+)/\d+\].*?'
            r'loss:\s*([\d.]+)'
        )
        
        # Pattern for individual losses
        loss_pattern = re.compile(r'(loss_\w+):\s*([\d.]+)')
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch, iter_num, total_loss = match.groups()
                    
                    entry = {
                        'epoch': int(epoch),
                        'iter': int(iter_num),
                        'loss': float(total_loss)
                    }
                    
                    # Extract individual losses
                    for loss_match in loss_pattern.finditer(line):
                        loss_name, loss_value = loss_match.groups()
                        entry[loss_name] = float(loss_value)
                    
                    entries.append(entry)
        
        return entries
    
    def verify(self) -> Dict:
        """Verify training losses.
        
        Returns:
            Verification report with:
                - stage: str
                - total_entries: int
                - found_losses: List[str]
                - missing_losses: List[str]
                - custom_loss_status: Dict[str, dict]
                - warnings: List[str]
                - passed: bool
        """
        entries = self.parse_log_file()
        
        if not entries:
            return {
                'stage': self.stage,
                'total_entries': 0,
                'found_losses': [],
                'missing_losses': self.expected_losses,
                'custom_loss_status': {},
                'warnings': ['No training entries found in log file'],
                'passed': False
            }
        
        # Find all loss keys that appear
        found_losses = set()
        for entry in entries:
            found_losses.update(k for k in entry.keys() if k.startswith('loss'))
        
        found_losses = sorted(found_losses)
        missing_losses = [l for l in self.expected_losses if l not in found_losses]
        
        # Analyze custom losses
        custom_loss_status = {}
        warnings = []
        
        for custom_loss in self.CUSTOM_LOSSES:
            if custom_loss not in self.expected_losses:
                continue  # Not expected in this stage
            
            if custom_loss in found_losses:
                values = self.loss_history[custom_loss]
                
                status = {
                    'found': True,
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'first': values[0],
                    'last': values[-1]
                }
                
                # Check for anomalies
                if all(v == 0 for v in values):
                    warnings.append(f"{custom_loss} is always 0.0 (possibly disabled)")
                    status['warning'] = 'always_zero'
                elif status['mean'] < 0.0001:
                    warnings.append(f"{custom_loss} mean={status['mean']:.6f} (very small)")
                    status['warning'] = 'very_small'
                
                custom_loss_status[custom_loss] = status
            else:
                custom_loss_status[custom_loss] = {
                    'found': False,
                    'count': 0
                }
                warnings.append(f"{custom_loss} NOT FOUND in logs")
        
        # Overall pass/fail
        passed = len(missing_losses) == 0
        
        return {
            'stage': self.stage,
            'total_entries': len(entries),
            'found_losses': found_losses,
            'missing_losses': missing_losses,
            'custom_loss_status': custom_loss_status,
            'warnings': warnings,
            'passed': passed
        }
    
    def print_report(self, report: Dict):
        """Print verification report."""
        print("=" * 70)
        print(f"Training Loss Verification Report")
        print("=" * 70)
        print(f"Log File: {self.log_file}")
        print(f"Stage: {report['stage']}")
        print(f"Total Entries: {report['total_entries']}")
        print()
        
        # Found losses
        print("Found Losses:")
        for loss in report['found_losses']:
            if loss in self.CUSTOM_LOSSES:
                print(f"  ✓ {loss} (custom)")
            else:
                print(f"  ✓ {loss}")
        print()
        
        # Missing losses
        if report['missing_losses']:
            print("Missing Losses:")
            for loss in report['missing_losses']:
                print(f"  ✗ {loss}")
            print()
        
        # Custom loss details
        if report['custom_loss_status']:
            print("Custom Loss Analysis:")
            for loss_name, status in report['custom_loss_status'].items():
                if status['found']:
                    print(f"\n  {loss_name}:")
                    print(f"    Count: {status['count']}")
                    print(f"    Mean:  {status['mean']:.6f}")
                    print(f"    Range: [{status['min']:.6f}, {status['max']:.6f}]")
                    print(f"    First: {status['first']:.6f}")
                    print(f"    Last:  {status['last']:.6f}")
                    
                    if 'warning' in status:
                        print(f"    ⚠️  Warning: {status['warning']}")
                else:
                    print(f"\n  {loss_name}: NOT FOUND")
            print()
        
        # Warnings
        if report['warnings']:
            print("Warnings:")
            for warning in report['warnings']:
                print(f"  ⚠️  {warning}")
            print()
        
        # Overall result
        print("=" * 70)
        if report['passed']:
            print("Result: ✓ PASSED - All expected losses found")
        else:
            print("Result: ✗ FAILED - Some expected losses missing")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify training losses in MMDetection logs'
    )
    parser.add_argument(
        'log_file',
        type=str,
        help='Path to training log file (*.log or *.log.json)'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='auto',
        choices=['auto', 'stage1', 'stage2', 'stage3'],
        help='Training stage (default: auto-detect from filename)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output report as JSON'
    )
    
    args = parser.parse_args()
    
    verifier = LossVerifier(args.log_file, args.stage)
    report = verifier.verify()
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        verifier.print_report(report)
    
    # Exit code: 0 if passed, 1 if failed
    exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
