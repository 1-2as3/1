"""全面深度检查：模型结构、参数、配置一致性"""
import sys
import torch
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS, DATASETS
from copy import deepcopy
import json

print("=" * 80)
print("深度模型检查：发现隐藏缺陷")
print("=" * 80)

register_all_modules(init_default_scope=True)

cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft.py')
base_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')

def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

merged = _deep_merge(deepcopy(base_cfg['model']), cfg.model)
roi = merged.get('roi_head', {})
if 'use_domain_loss' in roi:
    roi.pop('use_domain_loss')

# 构建模型
model = MODELS.build(merged)

print("\n" + "=" * 80)
print("检查 1: 配置文件一致性")
print("=" * 80)

issues = []

# 1.1 检查 load_from 路径
load_from = cfg.get('load_from', None)
print(f"\n1.1 预训练权重路径: {load_from}")
if load_from:
    import os
    if not os.path.exists(load_from):
        issues.append(f"❌ 预训练权重文件不存在: {load_from}")
        print(f"   ❌ 文件不存在，训练将从头开始！")
    else:
        print(f"   ✓ 文件存在")
else:
    issues.append("⚠️ 未设置 load_from，将从头训练（可能不符合 Stage2 预期）")
    print("   ⚠️ 未设置 load_from")

# 1.2 检查学习率与冻结策略
optim_cfg = cfg.get('optim_wrapper', {})
lr = optim_cfg.get('optimizer', {}).get('lr', None)
paramwise = optim_cfg.get('paramwise_cfg', {})
custom_keys = paramwise.get('custom_keys', {})

print(f"\n1.2 学习率与冻结策略:")
print(f"   基础学习率: {lr}")
if 'backbone' in custom_keys:
    bb_mult = custom_keys['backbone'].get('lr_mult', 1.0)
    print(f"   Backbone lr_mult: {bb_mult}")
    if bb_mult != 0.0:
        issues.append(f"⚠️ Backbone 未完全冻结 (lr_mult={bb_mult})，Stage2 应该冻结")
        print(f"   ⚠️ Backbone 未完全冻结！")
    else:
        print(f"   ✓ Backbone 已冻结")
else:
    issues.append("⚠️ 未设置 backbone 学习率倍率，默认不冻结")
    print("   ⚠️ 未配置 backbone 冻结")

# 1.3 检查训练轮数
max_epochs = cfg.get('train_cfg', {}).get('max_epochs', None)
if max_epochs is None:
    # 检查 scheduler
    scheduler = cfg.get('param_scheduler', {})
    if isinstance(scheduler, dict):
        max_epochs = scheduler.get('T_max', None)
print(f"\n1.3 训练轮数: {max_epochs}")
if max_epochs and max_epochs < 10:
    issues.append(f"⚠️ 训练轮数过少 ({max_epochs})，可能不足以收敛")
    print(f"   ⚠️ 轮数可能不足")

print("\n" + "=" * 80)
print("检查 2: 模型参数状态")
print("=" * 80)

# 2.1 统计所有参数
total_params = 0
trainable_params = 0
frozen_params = 0

param_groups = {
    'backbone': [],
    'neck': [],
    'rpn': [],
    'roi_head': [],
    'msp': [],
    'macl': [],
    'other': []
}

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    else:
        frozen_params += param.numel()
    
    # 分类参数
    if 'backbone' in name:
        param_groups['backbone'].append((name, param.numel(), param.requires_grad))
    elif 'neck' in name:
        if 'msp' in name.lower():
            param_groups['msp'].append((name, param.numel(), param.requires_grad))
        else:
            param_groups['neck'].append((name, param.numel(), param.requires_grad))
    elif 'rpn' in name:
        param_groups['rpn'].append((name, param.numel(), param.requires_grad))
    elif 'roi_head' in name:
        if 'macl' in name.lower():
            param_groups['macl'].append((name, param.numel(), param.requires_grad))
        else:
            param_groups['roi_head'].append((name, param.numel(), param.requires_grad))
    else:
        param_groups['other'].append((name, param.numel(), param.requires_grad))

print(f"\n2.1 参数总览:")
print(f"   总参数: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   可训练: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"   冻结的: {frozen_params:,} ({frozen_params/1e6:.2f}M)")

print(f"\n2.2 各模块参数分布:")
for group_name, params_list in param_groups.items():
    if not params_list:
        continue
    group_total = sum(p[1] for p in params_list)
    group_trainable = sum(p[1] for p in params_list if p[2])
    group_frozen = group_total - group_trainable
    print(f"\n   {group_name.upper()}:")
    print(f"     总计: {group_total:,} ({group_total/1e6:.2f}M)")
    print(f"     可训练: {group_trainable:,} ({group_trainable/1e6:.2f}M)")
    print(f"     冻结: {group_frozen:,} ({group_frozen/1e6:.2f}M)")
    
    # 检查异常
    if group_name == 'backbone':
        if group_trainable > 0:
            issues.append(f"❌ Backbone 有 {group_trainable:,} 个可训练参数，应该全部冻结！")
            print(f"     ❌ 发现可训练参数（应全部冻结）")
            # 列出前5个可训练的
            trainable = [p for p in params_list if p[2]][:5]
            for name, num, _ in trainable:
                print(f"        - {name}: {num:,}")
    
    elif group_name in ['msp', 'macl']:
        if group_trainable == 0:
            issues.append(f"❌ {group_name.upper()} 模块没有可训练参数！")
            print(f"     ❌ 无可训练参数（模块失效）")
    
    # 显示前3个参数示例
    if params_list:
        print(f"     示例参数:")
        for name, num, grad in params_list[:3]:
            grad_str = "✓可训练" if grad else "✗冻结"
            print(f"       - {name}: {num:,} ({grad_str})")

print("\n" + "=" * 80)
print("检查 3: 模块实例化状态")
print("=" * 80)

# 3.1 检查关键模块存在性
print(f"\n3.1 关键模块:")
checks = {
    'backbone': (hasattr(model, 'backbone'), type(model.backbone).__name__ if hasattr(model, 'backbone') else 'N/A'),
    'neck': (hasattr(model, 'neck'), type(model.neck).__name__ if hasattr(model, 'neck') else 'N/A'),
    'neck.msp_module': (hasattr(model.neck, 'msp_module') if hasattr(model, 'neck') else False, 
                        type(model.neck.msp_module).__name__ if (hasattr(model, 'neck') and hasattr(model.neck, 'msp_module')) else 'N/A'),
    'rpn_head': (hasattr(model, 'rpn_head'), type(model.rpn_head).__name__ if hasattr(model, 'rpn_head') else 'N/A'),
    'roi_head': (hasattr(model, 'roi_head'), type(model.roi_head).__name__ if hasattr(model, 'roi_head') else 'N/A'),
    'roi_head.macl_head': (hasattr(model.roi_head, 'macl_head') if hasattr(model, 'roi_head') else False,
                           type(model.roi_head.macl_head).__name__ if (hasattr(model, 'roi_head') and hasattr(model.roi_head, 'macl_head')) else 'N/A'),
}

for name, (exists, type_name) in checks.items():
    status = "✓" if exists else "✗"
    print(f"   {status} {name}: {type_name}")
    if not exists and '.' in name:
        parent = name.rsplit('.', 1)[0]
        issues.append(f"❌ {name} 不存在")

# 3.2 检查 MSP 和 MACL 的配置参数
if hasattr(model.neck, 'msp_module'):
    msp = model.neck.msp_module
    print(f"\n3.2 MSP 模块配置:")
    print(f"   channels: {getattr(msp, 'channels', 'N/A')}")
    print(f"   reduction: {getattr(msp, 'reduction', 'N/A')}")
    alpha_val = getattr(msp, 'alpha', None)
    if alpha_val is not None:
        if isinstance(alpha_val, torch.nn.Parameter):
            print(f"   alpha (可学习): 初始值={alpha_val.item():.4f}")
        else:
            print(f"   alpha (固定): {alpha_val}")

if hasattr(model.roi_head, 'macl_head'):
    macl = model.roi_head.macl_head
    print(f"\n3.3 MACL 模块配置:")
    print(f"   in_dim: {getattr(macl, 'in_dim', 'N/A')}")
    print(f"   proj_dim: {getattr(macl, 'proj_dim', 'N/A')}")
    tau_val = getattr(macl, 'tau', None)
    if tau_val is not None:
        if isinstance(tau_val, torch.nn.Parameter):
            print(f"   tau (可学习): 初始值={tau_val.item():.4f}")
        else:
            print(f"   tau (固定): {tau_val}")
    print(f"   use_dhn: {getattr(macl, 'use_dhn', 'N/A')}")
    if hasattr(macl, 'dhn_sampler') and macl.dhn_sampler:
        dhn = macl.dhn_sampler
        print(f"   DHN queue_size: {getattr(dhn, 'queue_size', 'N/A')}")
        print(f"   DHN momentum: {getattr(dhn, 'momentum', 'N/A')}")

print("\n" + "=" * 80)
print("检查 4: 数据集配置")
print("=" * 80)

# 4.1 检查训练/验证/测试集
for split_name in ['train', 'val', 'test']:
    split_key = f'{split_name}_dataloader'
    ds_cfg = None
    if split_key in cfg and cfg[split_key] is not None:
        if isinstance(cfg[split_key], dict):
            ds_cfg = cfg[split_key].get('dataset', {})
        else:
            ds_cfg = getattr(cfg[split_key], 'dataset', {}) if hasattr(cfg[split_key], 'dataset') else {}
    elif 'data' in cfg and split_name in cfg.data:
        ds_cfg = cfg.data[split_name]
    
    if not ds_cfg:
        print(f"\n4.{['train','val','test'].index(split_name)+1} {split_name.upper()} 数据集:")
        print(f"   ✗ 未配置")
        if split_name == 'train':
            issues.append(f"❌ 缺少训练集配置")
        continue
    
    print(f"\n4.{['train','val','test'].index(split_name)+1} {split_name.upper()} 数据集:")
    print(f"   type: {ds_cfg.get('type', 'N/A')}")
    print(f"   data_root: {ds_cfg.get('data_root', 'N/A')}")
    print(f"   ann_file: {ds_cfg.get('ann_file', 'N/A')}")
    print(f"   return_modality_pair: {ds_cfg.get('return_modality_pair', 'N/A')}")
    
    # 检查路径存在性
    import os
    data_root = ds_cfg.get('data_root', '')
    ann_file = ds_cfg.get('ann_file', '')
    if data_root and not os.path.exists(data_root):
        issues.append(f"❌ {split_name} 数据集 data_root 不存在: {data_root}")
        print(f"   ❌ data_root 不存在")
    if ann_file and not os.path.exists(ann_file):
        issues.append(f"❌ {split_name} 数据集 ann_file 不存在: {ann_file}")
        print(f"   ❌ ann_file 不存在")
    
    # 检查 return_modality_pair
    pair_mode = ds_cfg.get('return_modality_pair', False)
    if pair_mode:
        issues.append(f"⚠️ {split_name} 数据集启用了 return_modality_pair=True，这会跳过标准 pipeline")
        print(f"   ⚠️ 启用了配对模式（可能不兼容标准训练）")

print("\n" + "=" * 80)
print("检查 5: 损失函数配置")
print("=" * 80)

# 5.1 检查 RoI Head 的损失权重
if hasattr(model.roi_head, 'lambda1'):
    print(f"\n5.1 损失权重:")
    print(f"   lambda1 (MACL): {model.roi_head.lambda1}")
    print(f"   lambda2 (DHN): {model.roi_head.lambda2}")
    print(f"   lambda3 (Domain): {model.roi_head.lambda3}")
    
    if model.roi_head.lambda1 == 0:
        issues.append("⚠️ MACL 损失权重为0，模块将不参与训练")
        print(f"   ⚠️ lambda1=0，MACL 损失被禁用")

print("\n" + "=" * 80)
print("检查 6: 前向传播测试")
print("=" * 80)

# 6.1 创建模拟输入
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # 模拟输入
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 640).to(device)
    
    # 创建 data_samples
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    data_samples = []
    for i in range(batch_size):
        ds = DetDataSample()
        ds.set_metainfo({'img_shape': (512, 640), 'modality': 'infrared'})
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32).to(device)
        gt_instances.labels = torch.tensor([0], dtype=torch.long).to(device)
        ds.gt_instances = gt_instances
        data_samples.append(ds)
    
    print(f"\n6.1 推理测试 (batch_size={batch_size}):")
    with torch.no_grad():
        outputs = model(dummy_input, data_samples, mode='predict')
    print(f"   ✓ 推理成功，输出数量: {len(outputs)}")
    
    print(f"\n6.2 训练模式测试:")
    model.train()
    losses = model(dummy_input, data_samples, mode='loss')
    print(f"   ✓ 损失计算成功")
    print(f"   损失项:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"     - {k}: {v.item():.4f}")
        elif isinstance(v, (list, tuple)):
            print(f"     - {k}: {[x.item() if isinstance(x, torch.Tensor) else x for x in v]}")
    
    # 检查关键损失是否存在
    expected_losses = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
    for loss_name in expected_losses:
        if loss_name not in losses:
            issues.append(f"⚠️ 缺少预期损失项: {loss_name}")
            print(f"   ⚠️ 缺少 {loss_name}")
    
    # 检查是否有 MACL 相关损失
    macl_losses = [k for k in losses.keys() if 'macl' in k.lower()]
    if not macl_losses and hasattr(model.roi_head, 'macl_head'):
        issues.append("⚠️ MACL 模块存在但未产生损失")
        print(f"   ⚠️ 未检测到 MACL 损失（模块可能未激活）")
    elif macl_losses:
        print(f"   ✓ 检测到 MACL 损失: {macl_losses}")
    
except Exception as e:
    issues.append(f"❌ 前向传播测试失败: {e}")
    print(f"\n   ❌ 前向测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("问题汇总")
print("=" * 80)

if issues:
    print(f"\n发现 {len(issues)} 个问题:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("\n✓ 未发现明显问题")

print("\n" + "=" * 80)
print("建议修复优先级")
print("=" * 80)

critical = [i for i in issues if i.startswith('❌')]
warnings = [i for i in issues if i.startswith('⚠️')]

if critical:
    print(f"\n🔴 严重问题 ({len(critical)}):")
    for issue in critical:
        print(f"   {issue}")

if warnings:
    print(f"\n🟡 警告 ({len(warnings)}):")
    for issue in warnings:
        print(f"   {issue}")

print("\n" + "=" * 80)
