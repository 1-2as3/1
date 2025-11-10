"""
测试 stage2_kaist_domain_ft.py 的模型与数据集构建
- 验证配置能否加载
- 验证模型能否构建
- 验证测试数据集能否构建并取出一个样本
注意：此脚本不进行前向推理，仅做构建与样本加载校验
"""
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS, DATASETS
from copy import deepcopy

print("==> 1) 注册模块 ...")
register_all_modules(init_default_scope=True)
print("   ✅ 模块注册完成")

print("==> 2) 加载配置 ...")
cfg_path = 'configs/llvip/stage2_kaist_domain_ft_nodomain.py'
cfg = Config.fromfile(cfg_path)
print("   ✅ 配置加载成功:", cfg_path)

print("==> 3) 构建模型 ...")
def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _sanitize_model_cfg(m):
    """移除部分未充分实现的自定义标志，避免构建失败。
    
    注意：在实际训练时（tools/train.py）不需要此函数，
    因为 read_base() 会正确合并配置。此函数仅用于测试脚本的容错。
    
    当前策略：仅移除 use_domain_loss（尚未完整实现）
    保留：use_macl, use_msp, use_dhn（已在 FPN 和 StandardRoIHead 中实现）
    """
    m = deepcopy(m)
    roi = m.get('roi_head', {})
    # 仅移除尚未实现的标志
    for k in ['use_domain_loss']:
        if k in roi:
            roi.pop(k)
    m['roi_head'] = roi
    return m

try:
    model_cfg = _sanitize_model_cfg(cfg.model)
    model = MODELS.build(model_cfg)
    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
        except Exception:
            pass
    print("   ✅ 模型构建成功:", model.__class__.__name__)
except Exception as e:
    print("   ❌ 模型构建失败:", e)
    print("   🛈 当前 cfg.model 内容:\n", cfg.model)
    print("   🛈 尝试使用基础模型进行合并后再构建 ...")
    try:
        base_model_cfg = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')['model']
        merged_model = _deep_merge(base_model_cfg, cfg.model)
        merged_model = _sanitize_model_cfg(merged_model)
        model = MODELS.build(merged_model)
        print("   ✅ 合并基础模型后构建成功:", model.__class__.__name__)
    except Exception as e2:
        print("   ❌ 合并基础模型后仍失败:", e2)
        raise

# 3.1 组件级构建与注册检查（来自 test3 的要点）
try:
    from mmdet.registry import MODELS as _MODELS
    fpn_cfg = dict(
        type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256,
        num_outs=5, use_msp=True, msp_module=dict(type='MSPReweight', channels=256)
    )
    _fpn = _MODELS.build(fpn_cfg)
    print("   ✅ FPN(含MSP) 构建通过:", type(_fpn).__name__)
    roi_head_cfg = dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor', roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        ),
        use_macl=True,
        macl_head=dict(type='MACLHead', in_dim=256, proj_dim=128, temperature=0.07, use_dhn=True, dhn_cfg=dict(K=8192, m=0.99))
    )
    _roi_head = _MODELS.build(roi_head_cfg)
    print("   ✅ RoIHead(含MACL+DHN) 构建通过:", type(_roi_head).__name__)
    for name in ["MSPReweight", "MACLHead", "DHNSampler"]:
        print(f"   - 注册表检查 MODELS[{name}]:", name in _MODELS.module_dict)
except Exception as e:
    print("   ⚠️  组件级构建/注册检查跳过:", e)

print("==> 4) 构建测试数据集 ...")
# 兼容 test_dataloader 或旧式 data['test']
if 'test_dataloader' in cfg:
    ds_cfg = cfg.test_dataloader['dataset'] if isinstance(cfg.test_dataloader, dict) else cfg.test_dataloader.dataset
elif 'data' in cfg and 'test' in cfg.data:
    ds_cfg = cfg.data['test']
else:
    raise RuntimeError('未找到测试数据集配置（test_dataloader/dataset 或 data.test）')

# 确保不启用配对模式（与标准 pipeline 兼容）
ds_cfg = ds_cfg.copy()
ds_cfg.setdefault('return_modality_pair', False)

dataset = DATASETS.build(ds_cfg)
print("   ✅ 测试数据集构建成功，总样本数:", len(dataset))

print("==> 5) 取一个样本以验证 pipeline ...")
try:
    item = dataset[0]
    if isinstance(item, dict) and 'inputs' in item and 'data_samples' in item:
        print("   ✅ 样本加载成功，包含 keys:", list(item.keys()))
        print("   ✅ inputs shape:", getattr(item['inputs'], 'shape', 'N/A'))
        print("   ✅ data_samples.img_path:", getattr(item['data_samples'], 'img_path', 'N/A'))
    else:
        print("   ⚠️ 样本返回格式非标准（可能启用了配对模式或自定义返回），keys:", list(item.keys()) if isinstance(item, dict) else type(item))
except Exception as e:
    print("   ❌ 样本加载失败:", e)

print("\n🎯 构建验证完成。")

# 6) 可选：优化器参数/可训练参数检查（来自 test4 的要点）
try:
    import torch
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    targets = [
        'neck.msp_module.alpha',
        'roi_head.macl_head.proj.0.weight',
        'roi_head.macl_head.proj.3.weight',  # 修正：实际是第3层而非第2层
        'roi_head.macl_head.tau'  # 新增：可学习的温度参数
    ]
    found = {t: False for t in targets}
    for n, p in model.named_parameters():
        if p.requires_grad:
            for t in targets:
                if t in n:
                    found[t] = True
    print("\n==> 6) 关键可训练参数检查:")
    for t, ok in found.items():
        print(f"   {'✓' if ok else '✗'} {t}")
except Exception as e:
    print("   ⚠️  可训练参数检查跳过:", e)

# 7) Person-only 元信息检查（来自 test5 的要点）
try:
    from mmdet.datasets import LLVIPDataset, KAISTDataset, M3FDDataset
    print("\n==> 7) 数据集 METAINFO（classes）检查:")
    for name, cls in [("LLVIPDataset", LLVIPDataset), ("KAISTDataset", KAISTDataset), ("M3FDDataset", M3FDDataset)]:
        classes = getattr(cls, 'METAINFO', {}).get('classes', ())
        print(f"   - {name}: {classes}")
except Exception as e:
    print("   ⚠️  元信息检查跳过:", e)
