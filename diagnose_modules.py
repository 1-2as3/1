"""诊断模型中 MSP 和 MACL 模块是否存在"""
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from copy import deepcopy

print("=" * 80)
print("模型模块诊断")
print("=" * 80)

register_all_modules(init_default_scope=True)

cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft.py')
base = Config.fromfile('configs/_base_/models/faster_rcnn_r50_fpn.py')['model']

def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

merged = _deep_merge(deepcopy(base), cfg.model)

# 移除自定义开关
roi = merged.get('roi_head', {})
for k in ['use_macl', 'use_msp', 'use_dhn', 'use_domain_loss']:
    if k in roi:
        roi.pop(k)

print("\n1. 配置检查:")
print("   neck 配置:", merged.get('neck', {}))
print("   roi_head 配置 keys:", list(merged.get('roi_head', {}).keys()))

model = MODELS.build(merged)

print("\n2. 构建后的模型结构:")
print(f"   模型类型: {type(model).__name__}")
print(f"   是否有 neck: {hasattr(model, 'neck')}")
print(f"   是否有 roi_head: {hasattr(model, 'roi_head')}")

if hasattr(model, 'neck'):
    print(f"   neck 类型: {type(model.neck).__name__}")
    print(f"   neck 是否有 msp_module: {hasattr(model.neck, 'msp_module')}")
    if hasattr(model.neck, 'use_msp'):
        print(f"   neck.use_msp: {model.neck.use_msp}")

if hasattr(model, 'roi_head'):
    print(f"   roi_head 类型: {type(model.roi_head).__name__}")
    print(f"   roi_head 是否有 macl_head: {hasattr(model.roi_head, 'macl_head')}")
    if hasattr(model.roi_head, 'use_macl'):
        print(f"   roi_head.use_macl: {model.roi_head.use_macl}")

print("\n3. 参数名称扫描:")
msp_params = []
macl_params = []
other_params = []

for name, param in model.named_parameters():
    if 'msp' in name.lower():
        msp_params.append(name)
    elif 'macl' in name.lower():
        macl_params.append(name)
    elif 'neck' in name or 'roi_head' in name:
        other_params.append(name)

print(f"\n   MSP 相关参数 ({len(msp_params)}):")
for p in msp_params[:5]:
    print(f"     - {p}")
if len(msp_params) > 5:
    print(f"     ... 还有 {len(msp_params)-5} 个")

print(f"\n   MACL 相关参数 ({len(macl_params)}):")
for p in macl_params[:5]:
    print(f"     - {p}")
if len(macl_params) > 5:
    print(f"     ... 还有 {len(macl_params)-5} 个")

print(f"\n   Neck/RoI 其他参数 (前10个):")
for p in other_params[:10]:
    print(f"     - {p}")

print("\n4. 目标参数存在性检查:")
targets = [
    'neck.msp_module.alpha',
    'roi_head.macl_head.proj.0.weight',
    'roi_head.macl_head.proj.2.weight'
]
for t in targets:
    exists = any(t in name for name, _ in model.named_parameters())
    print(f"   {'✓' if exists else '✗'} {t}")

print("\n" + "=" * 80)
print("问题分析:")
print("=" * 80)

if not msp_params:
    print("❌ MSP 模块未被实例化")
    print("   原因: test_stage2_build.py 的 _sanitize_model_cfg 移除了 use_msp")
    print("   解决: 需要在配置中显式添加 neck.use_msp=True 和 neck.msp_module 配置")

if not macl_params:
    print("❌ MACL 模块未被实例化")
    print("   原因: test_stage2_build.py 的 _sanitize_model_cfg 移除了 use_macl")
    print("   解决: 需要在配置中显式添加 roi_head.use_macl=True 和 roi_head.macl_head 配置")

if not msp_params and not macl_params:
    print("\n💡 关键问题:")
    print("   sanitize 函数为了避免构建失败移除了自定义开关，")
    print("   但这同时也阻止了 MSP 和 MACL 模块的实例化。")
    print("   需要检查 FPN 和 StandardRoIHead 的实现是否正确处理这些开关。")
