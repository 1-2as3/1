"""
TensorBoard Logger Helper

- 包装 SummaryWriter，简化在训练外部脚本中记录自定义标量：alpha_l、loss_macl 等
- 可附加到研究验证阶段
"""
import argparse
from pathlib import Path


def safe_imports():
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True, None
    except Exception as e:
        return False, e


def main():
    parser = argparse.ArgumentParser(description='简单 TensorBoard 标量记录工具')
    parser.add_argument('--logdir', type=str, default='runs/research')
    parser.add_argument('--tag', type=str, default='alpha_l')
    parser.add_argument('--value', type=float, default=0.0)
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    ok, err = safe_imports()
    if not ok:
        print(f'[ERROR] 依赖缺失：{err}')
        print('请先安装 tensorboard: pip install tensorboard')
        raise SystemExit(1)

    from torch.utils.tensorboard import SummaryWriter
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_scalar(args.tag, args.value, args.step)
    writer.close()
    print(f'[OK] 写入 {args.tag}={args.value} @step={args.step} 到 {args.logdir}')


if __name__ == '__main__':
    main()
