"""
KAIST 数据集配置示例
展示如何在训练配置中使用单模态和双模态模式
"""

# ============================================================
# 示例 1: 单模态训练（标准 MMDetection 流程）
# ============================================================
_base_ = ['../_base_/default_runtime.py']

# 数据集配置
data_root = 'C:/KAIST_processed/'

# 单模态 - 红外图像训练
train_dataloader_infrared = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=f'{data_root}/ImageSets/train.txt',
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=False,
        return_modality_pair=False,  # 单模态模式
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# ============================================================
# 示例 2: 双模态配对加载（需要自定义训练循环）
# ============================================================

# 双模态数据加载器（不使用 pipeline，直接返回原始图像对）
train_dataloader_paired = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=f'{data_root}/ImageSets/train.txt',
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=True,  # 使用 test_mode 跳过 pipeline
        return_modality_pair=True,  # 启用双模态配对
    )
)

# ============================================================
# 示例 3: 双模态融合推理配置
# ============================================================

test_dataloader_paired = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='KAISTDataset',
        data_root=data_root,
        ann_file=f'{data_root}/ImageSets/test.txt',
        data_prefix=dict(sub_data_root=data_root),
        ann_subdir='Annotations',
        test_mode=True,
        return_modality_pair=True,  # 测试时返回双模态
    )
)

# ============================================================
# 使用说明
# ============================================================
"""
1. 单模态训练（推荐用于预训练）:
   - 设置 return_modality_pair=False
   - 使用标准 MMDetection pipeline
   - 可以正常使用 tools/train.py

2. 双模态配对加载（用于多模态融合）:
   - 设置 return_modality_pair=True
   - test_mode=True（跳过 pipeline）
   - 返回的数据格式:
     {
         'visible': np.ndarray,        # 可见光图像 (H, W, 3)
         'infrared': np.ndarray,       # 红外图像 (H, W, 3)
         'visible_path': str,          # 可见光路径
         'infrared_path': str,         # 红外路径
         'base_id': str,               # 样本基础ID
         'instances': List[dict],      # 标注实例
         'metainfo': dict              # 元信息（modality='paired'）
     }
   - 需要自定义训练循环处理双模态数据

3. 路径要求:
   - ann_file: 必须是完整绝对路径
   - data_prefix: 必须是 dict(sub_data_root='...')
   - 目录结构:
     C:/KAIST_processed/
     ├── Annotations/
     ├── visible/
     ├── infrared/
     └── ImageSets/{train,val,test}.txt

4. 样本ID命名规则:
   - 红外: set00_V000_lwir_I01216
   - 可见光: set00_V000_visible_I01216
   - Base ID: set00_V000_I01216（用于配对）
"""
