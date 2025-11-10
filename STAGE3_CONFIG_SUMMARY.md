r"""
===================================================================================
Stage3 联合训练配置完善总结
===================================================================================

配置文件：configs/llvip/stage3_joint_multimodal.py

===================================================================================
已完成的修改
===================================================================================

1. ✅ 添加损失权重配置
   - lambda_macl: 0.3 (MACL 多尺度对齐对比学习损失权重)
   - lambda_dhn: 0.5 (DHN 域协调网络损失权重)
   - lambda_domain: 0.2 (域对齐损失权重)
   
   注意：这些权重配置仅作为配置文件占位符，实际生效需要在 
   mmdet/models/roi_heads/standard_roi_head.py 中实现相应的损失计算逻辑。

2. ✅ 完善学习率调度器配置
   修改前：
   ```python
   param_scheduler = dict(
       type='CosineAnnealingLR', 
       eta_min=1e-5
   )
   ```
   
   修改后：
   ```python
   param_scheduler = dict(
       type='CosineAnnealingLR', 
       T_max=12,        # 余弦退火周期（epoch 数）
       eta_min=1e-6,    # 最小学习率
       by_epoch=True
   )
   ```

3. ✅ 修正优化器配置
   添加了 momentum 和 weight_decay 参数：
   ```python
   optim_wrapper = dict(
       optimizer=dict(
           type='SGD', 
           lr=0.0005,
           momentum=0.9,
           weight_decay=0.0001
       )
   )
   ```

4. ✅ 添加模型加载路径
   ```python
   load_from = './work_dirs/person_only_stage2/epoch_latest.pth'
   ```

5. ✅ 修正多数据集配置路径
   修改前：使用了错误的路径格式
   修改后：使用正确的 VOC 格式路径
   ```python
   data = dict(
       train=dict(
           type='ConcatDataset',
           datasets=[
               dict(
                   type='KAISTDataset', 
                   data_root='C:/KAIST_processed/',
                   ann_file='C:/KAIST_processed/ImageSets/train.txt',
                   data_prefix=dict(sub_data_root='C:/KAIST_processed'),
                   ann_subdir='Annotations',
                   return_modality_pair=False,
               ),
               dict(
                   type='M3FDDataset', 
                   data_root='C:/M3FD_processed/',
                   ann_file='C:/M3FD_processed/ImageSets/train.txt',
                   data_prefix=dict(sub_data_root='C:/M3FD_processed'),
                   ann_subdir='Annotations',
                   return_modality_pair=False,
               )
           ]
       )
   )
   ```

6. ✅ 修正测试数据集配置
   修改前：使用 LLVIPDataset，配置不完整
   修改后：使用 KAISTDataset，配置完整
   ```python
   test_dataloader = dict(
       batch_size=1,
       dataset=dict(
           type='KAISTDataset',
           data_root='C:/KAIST_processed/',
           ann_file='C:/KAIST_processed/ImageSets/test.txt',
           data_prefix=dict(sub_data_root='C:/KAIST_processed'),
           ann_subdir='Annotations',
           return_modality_pair=False,
           pipeline=test_pipeline,
       )
   )
   ```

===================================================================================
配置验证结果
===================================================================================

运行 test_stage3_config.py 后的验证结果：

✅ 配置加载成功
✅ 损失权重配置：lambda_macl=0.3, lambda_dhn=0.5, lambda_domain=0.2
✅ 学习率调度器：CosineAnnealingLR, T_max=12, eta_min=1e-6
✅ 优化器配置：SGD, lr=0.0005, momentum=0.9, weight_decay=0.0001
✅ 模型加载路径：./work_dirs/person_only_stage2/epoch_latest.pth
✅ 训练数据集：ConcatDataset (KAISTDataset + M3FDDataset)
✅ 测试数据集：KAISTDataset

⚠️  警告：
  - Stage2 权重文件尚不存在（需先完成 stage2 训练）
  - M3FD 数据集标注文件不存在（需先运行数据准备脚本）

===================================================================================
待完成事项
===================================================================================

1. 数据准备
   - ✅ KAIST 数据集已准备完成 (C:/KAIST_processed/)
   - ⚠️  M3FD 数据集需要准备：
     ```bash
     cd C:\Users\Xinyu\mmdetection\scripts
     python clean_and_split_m3fd_person.py
     ```

2. Stage2 训练
   在运行 stage3 之前，需要先完成 stage2 训练以生成预训练权重：
   ```bash
   cd C:\Users\Xinyu\mmdetection
   C:\Users\Xinyu\.conda\envs\py311\python.exe tools\train.py configs\llvip\stage2_kaist_domain_ft.py
   ```

3. 添加 Pipeline 配置
   当前 ConcatDataset 中的子数据集缺少 pipeline 配置。
   需要在配置文件开头添加：
   ```python
   # 训练 pipeline
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations', with_bbox=True, with_label=True),
       dict(type='Resize', scale=(640, 512), keep_ratio=True),
       dict(type='RandomFlip', prob=0.5),
       dict(type='PackDetInputs')
   ]
   ```
   
   然后在每个数据集配置中添加：
   ```python
   pipeline=train_pipeline,
   ```

4. 实现自定义损失（可选）
   如果需要使用 loss_cfg 中的权重，需要在 StandardRoIHead 中实现：
   - MACL 损失计算
   - DHN 损失计算
   - Domain Alignment 损失计算

===================================================================================
使用指南
===================================================================================

1. 验证配置
   ```bash
   cd C:\Users\Xinyu\mmdetection
   C:\Users\Xinyu\.conda\envs\py311\python.exe test_stage3_config.py
   ```

2. 准备数据（如果 M3FD 尚未准备）
   ```bash
   C:\Users\Xinyu\.conda\envs\py311\python.exe scripts\clean_and_split_m3fd_person.py
   ```

3. 完成 Stage2 训练（如果尚未完成）
   ```bash
   C:\Users\Xinyu\.conda\envs\py311\python.exe tools\train.py configs\llvip\stage2_kaist_domain_ft.py
   ```

4. 运行 Stage3 联合训练
   ```bash
   C:\Users\Xinyu\.conda\envs\py311\python.exe tools\train.py configs\llvip\stage3_joint_multimodal.py
   ```

===================================================================================
文件清单
===================================================================================

修改的文件：
  ✅ configs/llvip/stage3_joint_multimodal.py

新增的文件：
  ✅ test_stage3_config.py (配置验证脚本)

相关文件：
  - configs/llvip/stage2_kaist_domain_ft.py (Stage2 配置)
  - test_stage2_build.py (Stage2 构建验证)
  - mmdet/datasets/kaist_dataset.py (KAIST 数据集实现)
  - mmdet/datasets/m3fd_dataset.py (M3FD 数据集实现，待实现)

===================================================================================
配置参数说明
===================================================================================

学习率调度器参数：
  - T_max: 余弦退火周期，通常设置为训练的总 epoch 数
  - eta_min: 最小学习率，防止学习率降至 0
  - by_epoch: 按 epoch 调度（而非按 iteration）

损失权重参数（占位符，需在代码中实现）：
  - lambda_macl: 多尺度对齐对比学习损失权重（推荐 0.2-0.5）
  - lambda_dhn: 域协调网络损失权重（推荐 0.3-0.7）
  - lambda_domain: 域对齐损失权重（推荐 0.1-0.3）

优化器参数：
  - lr: 学习率，联合训练建议 0.0003-0.0005
  - momentum: SGD 动量，标准值 0.9
  - weight_decay: L2 正则化系数，标准值 0.0001

===================================================================================
常见问题
===================================================================================

Q1: 配置验证时提示 M3FD 标注文件不存在？
A1: 需要先运行 M3FD 数据准备脚本：
    python scripts/clean_and_split_m3fd_person.py

Q2: 训练时提示找不到 stage2 权重？
A2: 需要先完成 stage2 训练，或注释掉 load_from 配置从头训练。

Q3: ConcatDataset 报错缺少 pipeline？
A3: 需要在配置文件中添加 train_pipeline 定义，并在每个子数据集中引用。

Q4: 损失权重配置不生效？
A4: loss_cfg 仅作为配置占位符，需要在 StandardRoIHead 中实现相应损失计算。

===================================================================================

生成时间：2025-11-07
配置版本：Stage3 Joint Multimodal Training
验证状态：✅ PASSED

===================================================================================
"""

if __name__ == '__main__':
    print(__doc__)
