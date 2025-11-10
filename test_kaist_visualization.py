"""
KAIST 数据集加载与可视化测试脚本
功能：加载 KAIST 双模态配对数据并可视化验证
"""
import os
import cv2
import numpy as np
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS

print("=" * 80)
print("KAIST 数据集可视化测试")
print("=" * 80)

# 步骤 1: 注册 MMDetection 模块（必需！）
print("\n步骤 1: 注册模块...")
register_all_modules(init_default_scope=True)
print("✅ 注册完成")

# 步骤 2: 构建 KAIST 数据集（双模态配对模式）
print("\n步骤 2: 构建数据集...")
data_root = 'C:/KAIST_processed/'

# 配置：双模态配对加载
dataset_cfg = dict(
    type='KAISTDataset',
    data_root=data_root,
    ann_file=os.path.join(data_root, 'ImageSets/train.txt'),
    data_prefix=dict(sub_data_root=data_root),
    ann_subdir='Annotations',
    test_mode=True,  # test_mode 跳过 pipeline
    return_modality_pair=True,  # 启用双模态配对
)

dataset = DATASETS.build(dataset_cfg)
print(f"✅ 数据集构建成功")
print(f"   样本总数: {len(dataset)}")
print(f"   类别: {dataset.METAINFO['classes']}")

# 步骤 3: 加载并可视化前 3 对样本
print("\n步骤 3: 加载并可视化样本...")
output_dir = "kaist_visualization_output"
os.makedirs(output_dir, exist_ok=True)

for i in range(min(3, len(dataset))):
    try:
        # 加载双模态数据
        item = dataset[i]
        
        vis = item['visible']       # (H, W, 3) BGR
        ir = item['infrared']       # (H, W, 3) BGR
        base_id = item['base_id']
        instances = item['instances']
        
        print(f"\n样本 {i}:")
        print(f"  base_id: {base_id}")
        print(f"  visible shape: {vis.shape}")
        print(f"  infrared shape: {ir.shape}")
        print(f"  instances: {len(instances)} 个目标")
        
        # 在图像上绘制标注框
        vis_annotated = vis.copy()
        ir_annotated = ir.copy()
        
        for inst in instances:
            bbox = inst['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制边界框（红色）
            cv2.rectangle(vis_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(ir_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 添加类别标签
            label = 'person'
            cv2.putText(vis_annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(ir_annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 并排拼接：可见光 | 红外
        both = cv2.hconcat([vis_annotated, ir_annotated])
        
        # 添加标题栏
        title_height = 40
        title_bar = np.ones((title_height, both.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_bar, f"Visible (RGB)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_bar, f"Infrared (Thermal)", (both.shape[1]//2 + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 垂直拼接：标题栏 + 图像
        final_img = cv2.vconcat([title_bar, both])
        
        # 保存
        output_path = os.path.join(output_dir, f"sample_pair_{i}.jpg")
        cv2.imwrite(output_path, final_img)
        print(f"  ✅ 已保存: {output_path}")
        
        # 同时保存无标注版本（用于对比）
        both_clean = cv2.hconcat([vis, ir])
        clean_path = os.path.join(output_dir, f"sample_pair_{i}_clean.jpg")
        cv2.imwrite(clean_path, both_clean)
        print(f"  ✅ 已保存无标注版本: {clean_path}")
        
    except Exception as e:
        print(f"\n样本 {i}: ❌ 加载失败")
        print(f"  错误: {str(e)}")

# 步骤 4: 生成统计报告
print("\n" + "=" * 80)
print("✅ 可视化测试完成！")
print("=" * 80)
print(f"\n输出目录: {output_dir}/")
print(f"生成文件:")
print(f"  - sample_pair_0.jpg (带标注)")
print(f"  - sample_pair_0_clean.jpg (无标注)")
print(f"  - sample_pair_1.jpg")
print(f"  - sample_pair_1_clean.jpg")
print(f"  - sample_pair_2.jpg")
print(f"  - sample_pair_2_clean.jpg")
print("\n人工验证要点:")
print("  1. ✓ 可见光和红外图像是否为同一场景")
print("  2. ✓ 标注框是否正确标出行人位置")
print("  3. ✓ 两个模态的标注框是否一致")
print("  4. ✓ 图像质量是否清晰")
print("\n" + "=" * 80)

# 步骤 5: 额外测试 - 验证数据集完整性
print("\n[额外检查] 数据集完整性验证...")
sample_count = min(10, len(dataset))
valid_count = 0
error_count = 0

for i in range(sample_count):
    try:
        data = dataset[i]
        assert 'visible' in data, "缺少 visible 字段"
        assert 'infrared' in data, "缺少 infrared 字段"
        assert data['visible'].shape == data['infrared'].shape, "模态尺寸不一致"
        valid_count += 1
    except Exception as e:
        error_count += 1
        print(f"  样本 {i}: ⚠️  {str(e)}")

print(f"\n检查结果: {valid_count}/{sample_count} 样本有效")
if error_count == 0:
    print("✅ 数据集完整性检查通过！")
else:
    print(f"⚠️  发现 {error_count} 个问题样本")

print("\n" + "=" * 80)
print("测试脚本执行完毕")
print("=" * 80)
