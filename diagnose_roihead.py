"""详细诊断 RoIHead 配置传递"""
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from copy import deepcopy
import json

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

# 只移除 use_domain_loss
roi = merged.get('roi_head', {})
if 'use_domain_loss' in roi:
    roi.pop('use_domain_loss')

print("=" * 80)
print("RoIHead 配置详细检查")
print("=" * 80)

print("\n1. 原始配置 (cfg.model.roi_head):")
print(json.dumps(cfg.model.get('roi_head', {}), indent=2, default=str))

print("\n2. 合并后配置 (merged['roi_head']):")
roi_cfg = merged.get('roi_head', {})
print(f"   type: {roi_cfg.get('type')}")
print(f"   use_macl: {roi_cfg.get('use_macl')}")
print(f"   use_msp: {roi_cfg.get('use_msp')}")
print(f"   use_dhn: {roi_cfg.get('use_dhn')}")
print(f"   macl_head: {roi_cfg.get('macl_head')}")

print("\n3. 尝试构建 RoIHead:")
try:
    roi_head = MODELS.build(roi_cfg)
    print(f"   ✅ 构建成功: {type(roi_head).__name__}")
    print(f"   - roi_head.use_macl = {getattr(roi_head, 'use_macl', 'NOT_FOUND')}")
    print(f"   - hasattr(roi_head, 'macl_head') = {hasattr(roi_head, 'macl_head')}")
    if hasattr(roi_head, 'macl_head'):
        print(f"   - type(roi_head.macl_head) = {type(roi_head.macl_head).__name__}")
except Exception as e:
    print(f"   ❌ 构建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n4. 尝试构建完整模型:")
try:
    model = MODELS.build(merged)
    print(f"   ✅ 构建成功: {type(model).__name__}")
    print(f"   - model.roi_head.use_macl = {getattr(model.roi_head, 'use_macl', 'NOT_FOUND')}")
    print(f"   - hasattr(model.roi_head, 'macl_head') = {hasattr(model.roi_head, 'macl_head')}")
    
    # 参数扫描
    macl_params = [n for n, _ in model.named_parameters() if 'macl' in n.lower()]
    print(f"\n   MACL 参数数量: {len(macl_params)}")
    if macl_params:
        print("   MACL 参数列表:")
        for p in macl_params[:10]:
            print(f"     - {p}")
except Exception as e:
    print(f"   ❌ 构建失败: {e}")
    import traceback
    traceback.print_exc()
