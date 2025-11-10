"""
KAIST 数据集加载完整测试
验证：
1. 数据集构建成功
2. 批次加载正常
3. 数据格式正确
4. 标注信息完整
"""
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS
import torch
from torch.utils.data import DataLoader

print("=" * 80)
print("KAIST 数据集加载测试")
print("=" * 80)

# 注册模块
print("\n1. 注册模块...")
register_all_modules(init_default_scope=True)
print("   ✅ 完成")

# 加载配置
print("\n2. 加载配置...")
cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft_nodomain.py')
print("   ✅ 完成")

# 构建数据集
print("\n3. 构建测试数据集...")
if 'test_dataloader' in cfg:
    ds_cfg = cfg.test_dataloader['dataset'] if isinstance(cfg.test_dataloader, dict) else cfg.test_dataloader.dataset
else:
    raise RuntimeError("未找到 test_dataloader 配置")

ds_cfg = ds_cfg.copy()
ds_cfg.setdefault('return_modality_pair', False)

print("   正在构建数据集（可能需要几秒钟）...")
dataset = DATASETS.build(ds_cfg)
print(f"   ✅ 数据集构建成功，样本数: {len(dataset)}")

# 测试单个样本
print("\n4. 测试单个样本加载...")
try:
    sample = dataset[0]
    print("   ✅ 样本加载成功")
    print(f"      - keys: {list(sample.keys())}")
    if 'inputs' in sample:
        print(f"      - inputs shape: {sample['inputs'].shape}")
        print(f"      - inputs dtype: {sample['inputs'].dtype}")
    if 'data_samples' in sample:
        ds = sample['data_samples']
        print(f"      - img_path: {ds.img_path}")
        print(f"      - gt_instances: {len(ds.gt_instances)} 个目标")
        if hasattr(ds, 'metainfo') and 'modality' in ds.metainfo:
            print(f"      - modality: {ds.metainfo['modality']}")
except Exception as e:
    print(f"   ❌ 样本加载失败: {e}")
    raise

# 测试批次加载
print("\n5. 测试批次加载 (batch_size=2)...")
try:
    # 创建简单的 collate 函数
    from mmengine.dataset import pseudo_collate
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=pseudo_collate
    )
    
    batch = next(iter(loader))
    print("   ✅ 批次加载成功")
    print(f"      - batch type: {type(batch)}")
    print(f"      - batch length: {len(batch)}")
    
    # 检查批次中每个样本
    for i, item in enumerate(batch):
        if isinstance(item, dict):
            print(f"      - sample[{i}] keys: {list(item.keys())}")
            if 'inputs' in item:
                print(f"        inputs shape: {item['inputs'].shape}")
        
except Exception as e:
    print(f"   ❌ 批次加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试多个样本
print("\n6. 测试连续加载 10 个样本...")
success_count = 0
error_count = 0
for i in range(min(10, len(dataset))):
    try:
        sample = dataset[i]
        success_count += 1
    except Exception as e:
        error_count += 1
        print(f"   ⚠️  样本 {i} 加载失败: {e}")

print(f"   ✅ 成功: {success_count}/{10}, 失败: {error_count}/{10}")

# 检查数据分布
print("\n7. 检查数据统计信息...")
print(f"   - 数据集类型: {dataset.__class__.__name__}")
print(f"   - 总样本数: {len(dataset)}")
print(f"   - 类别: {dataset.METAINFO['classes']}")

# 统计目标数量（只检查前10个样本以加快速度）
total_instances = 0
samples_checked = min(10, len(dataset))
print(f"   正在统计前 {samples_checked} 个样本...")
for i in range(samples_checked):
    try:
        info = dataset.get_data_info(i)
        total_instances += len(info.get('instances', []))
    except:
        pass

avg_instances = total_instances / samples_checked if samples_checked > 0 else 0
print(f"   - 前 {samples_checked} 个样本平均目标数: {avg_instances:.2f}")

print("\n" + "=" * 80)
print("✅ KAIST 数据集加载测试完成")
print("=" * 80)
