"""
快速测试脚本：验证模型前向传播是否正常（无训练）
用于诊断anchor生成、数据加载等初始化问题
"""
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
import mmdet  # 必须import以注册所有模块
from mmdet.registry import DATASETS
import sys
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("操作超时！")

# 设置5分钟超时
if sys.platform != 'win32':
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

print("[1/5] 加载配置...")
cfg = Config.fromfile('configs/llvip/stage1_llvip_pretrain.py')

print("[2/5] 构建数据集...")
cfg.train_dataloader.dataset.pipeline = cfg.train_dataloader.dataset.pipeline[:3]  # 只保留前3步
dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f"     数据集大小: {len(dataset)}")

print("[3/5] 获取第一个样本...")
data = dataset[0]
print(f"     数据keys: {list(data.keys())}")
if 'inputs' in data:
    print(f"     输入shape: {data['inputs'].shape}")
else:
    print(f"     [WARN] 数据格式: {type(data)}")
    # 跳过完整pipeline的数据样本测试
    print("     跳过样本检查，直接测试模型...")

print("[4/5] 构建模型...")
model = MODELS.build(cfg.model)
model = model.cuda() if torch.cuda.is_available() else model
model.eval()
print(f"     模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

print("[5/5] 测试前向传播（使用随机输入）...")
with torch.no_grad():
    # 使用随机tensor测试
    inputs = torch.randn(1, 3, 640, 640)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # 测试extract_feat（最容易卡住的地方）
    print("     > extract_feat...")
    try:
        x = model.extract_feat(inputs)
        print(f"       特征提取成功! 输出层数: {len(x) if isinstance(x, (list, tuple)) else 'dict'}")
    except Exception as e:
        print(f"       ❌ 特征提取失败: {e}")
        sys.exit(1)
    
    # 测试RPN forward
    print("     > rpn_head.forward...")
    try:
        rpn_out = model.rpn_head(x)
        print(f"       RPN前向成功! 输出: {len(rpn_out)} tensors")
    except Exception as e:
        print(f"       ❌ RPN前向失败: {e}")
        sys.exit(1)

print("\n✅ 所有测试通过！模型前向传播正常。")
print("   可以尝试启动完整训练。")
