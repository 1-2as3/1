"""
Loss Detail Check

- 复用 scripts/check_training_losses.py 的能力，面向 tools/ 入口
- 输入 log 路径和 stage，输出详细分项并返回码指示是否通过
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='训练日志 Loss 分项检查')
    parser.add_argument('log_file', type=str, help='*.log 或 *.log.json')
    parser.add_argument('--stage', type=str, default='auto', choices=['auto', 'stage1', 'stage2', 'stage3'])
    parser.add_argument('--json', action='store_true', help='以 JSON 输出')
    args = parser.parse_args()

    # 动态导入脚本
    import importlib.util
    script_path = Path('scripts/check_training_losses.py')
    if not script_path.exists():
        print('[ERROR] 找不到 scripts/check_training_losses.py')
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location('check_training_losses', str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    verifier = mod.LossVerifier(args.log_file, args.stage)
    report = verifier.verify()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        verifier.print_report(report)

    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
