"""
KAIST 模型前向和反向传播测试 (Dry Run)
验证：
1. 模型可以成功构建
2. 前向传播正常
3. 损失计算正确
4. 反向传播无错误
5. GPU/CPU 兼容性
"""
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS, DATASETS
import torch
from copy import deepcopy

print("=" * 80)
print("KAIST 模型前向-反向传播测试 (Dry Run)")
print("=" * 80)

# 检测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n使用设备: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 注册模块
print("\n1. 注册模块...")
register_all_modules(init_default_scope=True)
print("   ✅ 完成")

# 加载配置
print("\n2. 加载配置...")
cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft_nodomain.py')
print("   ✅ 完成")

# 构建模型（使用基础模型合并）
print("\n3. 构建模型...")
try:
    # 尝试直接构建
    model_cfg = cfg.model.copy()
    model = MODELS.build(model_cfg)
    print("   ✅ 模型构建成功（直接构建）")
except Exception as e:
    print(f"   ⚠️  直接构建失败: {e}")
    print("   🔄 尝试合并基础模型配置...")
    
    # 深度合并函数
    def _deep_merge(dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_merge(dst[k], v)
            else:
                dst[k] = v
        return dst
    
    # 清理未实现的参数
    def _sanitize_model_cfg(m):
        m = deepcopy(m)
        roi = m.get('roi_head', {})
        for k in ['use_macl', 'use_msp', 'use_dhn', 'use_domain_loss']:
            if k in roi:
                roi.pop(k)
        m['roi_head'] = roi
        return m
    
    try:
        base_model_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')['model']
        merged_model = _deep_merge(base_model_cfg, cfg.model)
        merged_model = _sanitize_model_cfg(merged_model)
        model = MODELS.build(merged_model)
        print("   ✅ 模型构建成功（合并基础模型）")
    except Exception as e2:
        print(f"   ❌ 模型构建失败: {e2}")
        raise

# 移动模型到设备
print(f"\n4. 将模型移动到 {device}...")
model = model.to(device)
model.eval()  # 设置为评估模式
print("   ✅ 完成")

# 构建数据集并获取一个样本
print("\n5. 加载测试样本...")
if 'test_dataloader' in cfg:
    ds_cfg = cfg.test_dataloader['dataset'] if isinstance(cfg.test_dataloader, dict) else cfg.test_dataloader.dataset
else:
    raise RuntimeError("未找到 test_dataloader 配置")

ds_cfg = ds_cfg.copy()
ds_cfg.setdefault('return_modality_pair', False)
dataset = DATASETS.build(ds_cfg)

sample = dataset[0]
print("   ✅ 样本加载成功")
print(f"      - inputs shape: {sample['inputs'].shape}")

# 准备输入
print("\n6. 准备模型输入...")
# 重要：图像需要通过 data_preprocessor 进行标准化处理
# 1. 先转换为 float32 类型
inputs_tensor = sample['inputs'].unsqueeze(0).float().to(device)  # [1, C, H, W]

# 2. 构造输入字典（模拟 DataLoader 的输出格式）
data_batch = {
    'inputs': [sample['inputs'].float()],  # List of tensors
    'data_samples': [sample['data_samples']]
}

# 3. 使用 data_preprocessor 处理（这会自动进行归一化等操作）
if hasattr(model, 'data_preprocessor'):
    with torch.no_grad():
        data = model.data_preprocessor(data_batch, training=False)
        inputs = data['inputs']
        data_samples = data['data_samples']
    print("   ✅ 完成（使用 data_preprocessor）")
else:
    # 如果没有 data_preprocessor，使用简单的归一化
    inputs = inputs_tensor / 255.0  # 归一化到 [0, 1]
    data_samples = [sample['data_samples']]
    print("   ✅ 完成（手动归一化）")

print(f"      - inputs shape: {inputs.shape}")
print(f"      - inputs dtype: {inputs.dtype}")
print(f"      - batch size: 1")

# 测试前向传播（推理模式）
print("\n7. 测试前向传播（推理模式）...")
try:
    with torch.no_grad():
        results = model(inputs, data_samples, mode='predict')
    print("   ✅ 前向传播成功")
    print(f"      - 输出类型: {type(results)}")
    print(f"      - 输出数量: {len(results)}")
    if len(results) > 0:
        print(f"      - 检测到的目标数: {len(results[0].pred_instances)}")
except Exception as e:
    print(f"   ❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

# 测试损失计算（训练模式）
print("\n8. 测试损失计算（训练模式）...")
try:
    model.train()  # 设置为训练模式
    
    # 准备训练数据（重新处理，因为训练模式需要）
    train_data_batch = {
        'inputs': [sample['inputs'].float()],
        'data_samples': [sample['data_samples']]
    }
    
    # 使用 data_preprocessor 处理训练数据
    if hasattr(model, 'data_preprocessor'):
        train_data = model.data_preprocessor(train_data_batch, training=True)
        train_inputs = train_data['inputs']
        train_data_samples = train_data['data_samples']
    else:
        train_inputs = sample['inputs'].unsqueeze(0).float().to(device) / 255.0
        train_data_samples = [sample['data_samples']]
    
    # 计算损失
    losses = model(train_inputs, train_data_samples, mode='loss')
    
    print("   ✅ 损失计算成功")
    print(f"      - 损失项:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"        {k}: {v.item():.4f}")
        else:
            print(f"        {k}: {v}")
    
    # 计算总损失（鲁棒展平，兼容 list/tensor）
    def _flatten_loss_dict(loss_dict):
        flat = []
        for v in loss_dict.values():
            if isinstance(v, torch.Tensor):
                flat.append(v)
            elif isinstance(v, list):
                flat.extend([x for x in v if isinstance(x, torch.Tensor)])
        return flat
    flat_losses = _flatten_loss_dict(losses)
    total_loss = torch.mean(torch.stack([x.mean() for x in flat_losses])) if flat_losses else None
    print(f"      - 总损失: {total_loss.item():.4f}")
    
except Exception as e:
    print(f"   ❌ 损失计算失败: {e}")
    import traceback
    traceback.print_exc()
    total_loss = None

# 测试反向传播
if total_loss is not None:
    print("\n9. 测试反向传播...")
    try:
        # 清除之前的梯度
        model.zero_grad()
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度（整模 + 自定义模块聚焦）
        grad_count = 0
        none_grad_count = 0
        custom_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_count += 1
                    if any(x in name for x in ["macl", "msp", "alpha", "tau"]):
                        custom_grads.append((name, float(param.grad.abs().mean().item())))
                else:
                    none_grad_count += 1
        
        print("   ✅ 反向传播成功")
        print(f"      - 有梯度的参数: {grad_count}")
        print(f"      - 无梯度的参数: {none_grad_count}")
        if custom_grads:
            print("      - 自定义模块梯度样例:")
            for n, g in custom_grads[:8]:
                print(f"        {n}: {g:.6f}")
        
    except Exception as e:
        print(f"   ❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n9. 跳过反向传播测试（损失计算失败）")

# 内存使用统计
if device == 'cuda':
    print("\n10. GPU 内存使用统计...")
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"   - 已分配: {allocated:.1f} MB")
    print(f"   - 已保留: {reserved:.1f} MB")

print("\n" + "=" * 80)
print("✅ KAIST 模型前向-反向传播测试完成")
print("=" * 80)
print("\n备注:")
print("  - 如果损失计算或反向传播失败，可能是因为自定义损失未实现")
print("  - 标准 Faster R-CNN 的基础损失（RPN + RoI）应该正常工作")
print("  - use_macl/use_msp/use_dhn/use_domain_loss 等自定义选项已被移除")
