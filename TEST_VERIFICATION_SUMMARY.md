# 测试/检查/验证/监控/记录代码文件总结
**生成日期**: 2025-11-05  
**项目**: MMDetection 3.x + 多模态目标检测 (LLVIP/KAIST/M3FD)

---

## 📋 目录

1. [快速测试脚本 (根目录)](#1-快速测试脚本)
2. [训练监控 Hooks](#2-训练监控-hooks)
3. [数据集工具](#3-数据集工具)
4. [分析工具](#4-分析工具)
5. [单元测试](#5-单元测试)
6. [验证脚本](#6-验证脚本)
7. [文档报告](#7-文档报告)

---

## 1. 精简后的快速测试脚本（推荐）

### 📁 位置: `C:\Users\Xinyu\mmdetection\`

| 文件名 | 目的 | 关键功能 | 说明 |
|--------|------|---------|------|
| **verify_all.py** | 一键全链路验证 | 顺序运行构建/数据/前向三大检查；非交互倒计时启动 | 推荐首选 |
| **test_stage2_build.py** | 配置与模型构建 | 深度合并 base 配置的回退逻辑；构建 FasterRCNN，校验 num_classes=1 | 训练前必跑 |
| **test_dataset_kaist.py** | KAIST 数据加载 | 构建数据集与 DataLoader；样本/批次检查；统计信息 | 数据完整性 |
| **test_forward_kaist.py** | 前向/损失/反向 | 通过 data_preprocessor 保证 dtype/归一化；loss/backward 成功 | 数值稳定性 |
| **test_kaist_visualization.py** | 可视化样例 | 取若干样本绘制可视化，保存到本地 | 可选 |
| **test_module_switches.py** | 模块开关检查 | 验证 MACL/MSP/DHN 等开关是否按配置生效 | 可选 |
| **test_stage3_config.py** | Stage3 配置检查 | 学习率调度与损失权重覆盖项是否生效 | 可选 |

---

## 2. 训练监控 Hooks

### 📁 位置: `mmdet/engine/hooks/`

| Hook 名称 | 文件 | 监控/检查内容 | 触发时机 | 优先级 |
|----------|------|--------------|---------|--------|
| **MetricsExportHook** | `metrics_export_hook.py` | • Loss/mAP/梯度范数<br>• CSV 导出<br>• PNG 曲线图<br>• HTML 交互报告<br>• TensorBoard 日志<br>• Config 快照<br>• 自动压缩 ZIP | after_train_epoch<br>after_train | NORMAL |
| **TSNEVisualHook** | `tsne_visual_hook.py` | • MACL embedding 可视化<br>• t-SNE 2D 散点图<br>• 模态对齐指标<br>• Inter/Intra-modal 距离 | after_train_epoch | NORMAL |
| **CheckInvalidLossHook** | `checkloss_hook.py` | • Loss 是否为 NaN/Inf<br>• 每 N 个 iter 检查<br>• 异常时报错 | after_train_iter | NORMAL |
| **NumClassCheckHook** | `num_class_check_hook.py` | • 检查模型 num_classes<br>• 与数据集类别数匹配<br>• 配置一致性验证 | before_train | VERY_HIGH |
| **ParameterMonitorHook** | `parameter_monitor_hook.py` | • 参数值变化<br>• 梯度大小<br>• TensorBoard 记录<br>• MSP α 监控 | after_train_epoch<br>after_train_iter | NORMAL |
| **VisualizationHook** | `visualization_hook.py` | • 推理结果可视化<br>• 检测框绘制<br>• 保存到本地/TensorBoard | after_val_iter<br>after_test_iter | NORMAL |
| **MemoryProfilerHook** | `memory_profiler_hook.py` | • GPU 显存占用<br>• CPU 内存占用<br>• 内存泄漏检测 | after_train_iter | NORMAL |
| **SyncNormHook** | `sync_norm_hook.py` | • 多 GPU BatchNorm 同步<br>• 分布式训练一致性 | before_train_epoch | NORMAL |
| **DomainAdaptationHook** | `domain_adaptation_hook.py` | • 域对齐损失权重<br>• 逐步增加策略<br>• Stage 2/3 专用 | before_train_epoch<br>after_train_iter | NORMAL |

### 📊 Hooks 功能矩阵

| 功能类别 | 相关 Hook | 输出格式 |
|---------|-----------|---------|
| **损失监控** | MetricsExportHook, CheckInvalidLossHook | CSV, PNG, TensorBoard |
| **可视化** | TSNEVisualHook, VisualizationHook | PNG, HTML, TensorBoard |
| **参数监控** | ParameterMonitorHook | TensorBoard |
| **资源监控** | MemoryProfilerHook | 日志, TensorBoard |
| **配置检查** | NumClassCheckHook | 日志, 报错 |
| **域适应** | DomainAdaptationHook | 日志, 权重调整 |

---

## 3. 数据集工具

### 📁 位置: `tools/`

| 工具文件 | 功能 | 输出 | 使用场景 |
|---------|------|------|---------|
| **gen_dataset_report.py** | • 数据集统计分析<br>• 配对完整性检查<br>• 图像尺寸分布<br>• 样本数量分布<br>• ImageSets 生成 | `analysis_report/`<br>• pair_completeness_bar.png<br>• image_size_hist.png<br>• sample_distribution.png<br>• summary.json<br>• train.txt / val.txt | 数据准备阶段 |
| **browse_dataset.py** | • 可视化数据样本<br>• 检查标注正确性<br>• 交互式浏览 | 图像窗口显示 | 数据验证 |
| **export_training_metrics.py** | • 导出训练日志<br>• Loss/mAP 曲线<br>• 格式转换 | CSV, JSON | 实验分析 |

### 🔍 gen_dataset_report.py 详细功能

支持数据集: **LLVIP, KAIST, M3FD**

**输出报告**:
```
analysis_report/
├── pair_completeness_bar.png    # 可见光-红外配对完整性
├── image_size_hist.png           # 图像尺寸分布直方图
├── sample_distribution.png       # 训练/验证集分布
└── summary.json                  # 统计摘要
```

**检查项**:
- ✓ 可见光/红外图像配对完整性
- ✓ 缺失文件检测
- ✓ 图像尺寸统计
- ✓ 标注文件验证
- ✓ 训练/验证集划分 (80/20)

**使用示例**:
```bash
python tools/gen_dataset_report.py --dataset LLVIP --data-root C:/LLVIP/LLVIP
python tools/gen_dataset_report.py --dataset KAIST --split-ratio 0.8
python tools/gen_dataset_report.py --dataset M3FD --output-dir custom_report
```

---

## 4. 分析工具

### 📁 位置: `tools/analysis_tools/`

| 工具文件 | 功能 | 应用场景 |
|---------|------|---------|
| **analyze_logs.py** | 解析训练日志，提取 Loss/mAP | 实验对比 |
| **analyze_results.py** | 分析推理结果 JSON | 错误分析 |
| **confusion_matrix.py** | 生成混淆矩阵 | 分类错误分析 |
| **coco_error_analysis.py** | COCO 指标细分 | 定位性能瓶颈 |
| **eval_metric.py** | 评估指标计算 | 自定义指标 |
| **get_flops.py** | 计算模型 FLOPs/参数量 | 模型复杂度 |
| **robustness_eval.py** | 鲁棒性评估（噪声/遮挡） | 鲁棒性测试 |
| **test_robustness.py** | 对抗鲁棒性测试 | 安全性评估 |
| **benchmark.py** | 推理速度基准测试 | 性能对比 |
| **optimize_anchors.py** | 优化 Anchor 尺寸 | 模型调优 |

### 🎯 关键工具使用示例

#### **analyze_logs.py** - 日志分析
```bash
python tools/analysis_tools/analyze_logs.py \
  plot_curve \
  work_dirs/stage1_llvip_pretrain/20251105_*.log.json \
  --keys loss_total loss_macl \
  --legend stage1_total stage1_macl
```

#### **get_flops.py** - 模型复杂度
```bash
python tools/analysis_tools/get_flops.py \
  configs/llvip/stage1_llvip_pretrain.py \
  --shape 640 512
```

#### **confusion_matrix.py** - 混淆矩阵
```bash
python tools/analysis_tools/confusion_matrix.py \
  configs/llvip/stage2_kaist_domain_ft.py \
  work_dirs/stage2/results.pkl \
  --show --out confusion.png
```

---

## 5. 单元测试

### 📁 位置: `tests/`

| 测试类别 | 测试文件示例 | 测试内容 |
|---------|-------------|---------|
| **数据集测试** | `test_datasets/test_coco.py`<br>`test_datasets/test_pascal_voc.py` | 数据加载<br>标注解析<br>Pipeline 转换 |
| **数据变换测试** | `test_datasets/test_transforms/test_loading.py`<br>`test_datasets/test_transforms/test_geometric.py` | 图像加载<br>几何变换<br>颜色增强 |
| **模型测试** | `test_models/test_detectors/test_detr.py`<br>`test_models/test_roi_heads/test_cascade_roi_head.py` | 前向传播<br>损失计算<br>维度检查 |
| **损失测试** | `test_models/test_losses/test_loss.py` | 损失函数<br>梯度计算 |
| **Hook 测试** | `test_engine/test_hooks/test_checkloss_hook.py`<br>`test_engine/test_hooks/test_visualization_hook.py` | Hook 触发<br>功能验证 |
| **结构测试** | `test_structures/test_det_data_sample.py`<br>`test_structures/test_bbox/` | 数据结构<br>Bbox 操作 |
| **评估测试** | `test_evaluation/test_metrics/test_coco_metric.py` | mAP 计算<br>指标评估 |

### 🧪 测试执行

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_models/test_roi_heads/

# 运行单个测试文件
pytest tests/test_engine/test_hooks/test_checkloss_hook.py

# 带覆盖率报告
pytest --cov=mmdet tests/
```

---

## 6. 验证脚本

### 📁 位置: `tools/misc/`

| 脚本文件 | 验证内容 | 输出 |
|---------|---------|------|
| **verify_single_class_pipeline.py** | • 数据集 METAINFO (person-only)<br>• 配置文件 num_classes=1<br>• 模型构建<br>• 前向传播<br>• 推理模式 | 5 项测试报告<br>✓/✗ 状态 |

### 📋 verify_single_class_pipeline.py 详细测试

**Test 1**: Dataset METAINFO 检查
- LLVIPDataset: `classes=('person',)`
- KAISTDataset: `classes=('person',)`
- M3FDDataset: `classes=('person',)`

**Test 2**: Config num_classes 检查
- Stage 1: `num_classes=1`
- Stage 2: `num_classes=1`
- Stage 3: `num_classes=1`

**Test 3**: 模型构建验证
- FasterRCNN 实例化
- bbox_head.num_classes=1
- 参数量统计

**Test 4**: 前向传播验证
- 损失计算
- 损失项检查
- 数值有效性

**Test 5**: 推理模式验证
- 预测输出
- 类别标签检查 (all=0)
- Bbox 数量统计

**运行**:
```bash
python tools/misc/verify_single_class_pipeline.py
```

**输出示例**:
```
======================================================================
Person-Only Detection Pipeline Verification
======================================================================

[Test 1] Verifying Dataset METAINFO
----------------------------------------------------------------------
✓ LLVIPDataset        : classes=('person',) [OK]
✓ KAISTDataset        : classes=('person',) [OK]
✓ M3FDDataset         : classes=('person',) [OK]

✓ All datasets are person-only!

[Test 2] Verifying Config Files
----------------------------------------------------------------------
✓ Stage 1 (LLVIP)     : num_classes=1 [OK]
✓ Stage 2 (KAIST)     : num_classes=1 [OK]
✓ Stage 3 (Joint)     : num_classes=1 [OK]

✓ All configs are correctly set to num_classes=1!

... (完整报告)
```

---

## 7. 文档报告

### 📁 位置: 根目录

| 文档文件 | 内容 | 用途 |
|---------|------|------|
| **PERSON_ONLY_MIGRATION.md** | Person-only 迁移完整记录<br>• 修改清单<br>• 验证结果<br>• 训练命令<br>• 故障排查 | 迁移参考文档 |
| **MODULE_SWITCHES_ENHANCEMENT_REPORT.md** | 模块开关增强报告<br>• MACL/MSP/DHN 独立控制<br>• 配置示例<br>• 测试结果 | 功能说明文档 |
| **PAIRED_MODALITY_IMPLEMENTATION.md** | 成对模态实现报告<br>• Visible+Infrared 配对<br>• DataLoader 修改<br>• 模态标签传递 | 功能说明文档 |
| **ROBUST_LOSS_AGGREGATION_FIX.md** | 鲁棒损失聚合修复<br>• 混合类型处理<br>• NaN 防护<br>• 测试案例 | 调试参考文档 |

---

## 📊 测试/验证代码统计

### 按类别分类

| 类别 | 文件数量 | 主要位置 |
|------|---------|---------|
| **快速测试脚本** | 12 | 根目录 |
| **训练监控 Hooks** | 9 | `mmdet/engine/hooks/` |
| **数据集工具** | 3 | `tools/` |
| **分析工具** | 12 | `tools/analysis_tools/` |
| **单元测试** | 404+ | `tests/` |
| **验证脚本** | 1 | `tools/misc/` |
| **文档报告** | 4 | 根目录 |
| **总计** | **445+** | - |

### 按功能分类

| 功能 | 相关文件数 | 覆盖范围 |
|------|-----------|---------|
| **环境/注册检查** | 8 | 模块注册、版本验证 |
| **前向/反向传播** | 15 | 梯度流、损失计算 |
| **模型构建验证** | 12 | 参数检查、维度验证 |
| **数据集验证** | 25 | 配对检查、标注验证 |
| **训练监控** | 9 | Loss/mAP、参数、资源 |
| **可视化分析** | 8 | t-SNE、检测框、曲线图 |
| **性能分析** | 6 | FLOPs、速度、鲁棒性 |
| **错误诊断** | 12 | NaN 检测、混淆矩阵 |

---

## 🎯 推荐使用流程

### 阶段 1: 一键全链路验证（推荐）
```bash
python verify_all.py
```
说明：脚本会在 3 秒倒计时后自动开始，依次执行模型构建、数据加载、前向/损失/反向三大检查，并在异常时给出明确报错与排查建议。

### 阶段 2: 分步排查（如需单独定位）
```bash
python test_stage2_build.py     # 配置与模型构建（含 base 合并回退）
python test_dataset_kaist.py    # KAIST 数据/管道/批处理检查
python test_forward_kaist.py    # 前向/损失/反向（经 data_preprocessor）
```

### 阶段 3: 可选扩展
```bash
python test_kaist_visualization.py  # 小样本可视化
python test_module_switches.py      # 模块开关验证
python test_stage3_config.py        # Stage3 配置项检查
```

### 阶段 4: 训练监控
**在配置文件中启用**:
```python
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    tsne_visual=dict(type='TSNEVisualHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50),
    parameter_monitor=dict(type='ParameterMonitorHook', interval=1)
)
```

### 阶段 5: 训练后分析
```bash
1. python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/*/log.json
2. python tools/analysis_tools/confusion_matrix.py <config> <results.pkl>
3. python tools/analysis_tools/get_flops.py <config>
```

---

## 🔧 关键脚本说明

### MetricsExportHook - 综合监控 Hook

**自动生成**:
- `metrics_logs/run_YYYYMMDD_HHMMSS/`
  - `metrics.csv` - 完整训练数据
  - `metrics_curves.png` - Loss/mAP 曲线
  - `metrics_report.html` - 交互式报告
  - `config_snapshot_epoch{N}.py` - 配置快照
  - `tensorboard_logs/` - TensorBoard 事件
  - `run_*.zip` - 自动压缩包

**监控指标**:
- Loss: total, rpn_cls, rpn_bbox, cls, bbox, macl
- Gradient: grad_norm
- mAP (if available)

### TSNEVisualHook - Embedding 可视化

**生成内容**:
- `tsne_vis/tsne_epoch{N}.png`
  - 蓝色: Visible embedding
  - 红色: Infrared embedding
  - 显示模态对齐效果

**对齐指标**:
- Inter-modal distance: 模态间距离
- Intra-modal distance: 模态内距离
- Alignment score = inter / (vis_intra + ir_intra)
  - Score <1: 对齐成功 ✓
  - Score >1: 对齐不足

---

## 📈 监控可视化示例

### TensorBoard 查看
```bash
tensorboard --logdir work_dirs/stage1_llvip_pretrain/tensorboard_logs
# 访问 http://localhost:6006
```

**可视化内容**:
- Loss/total, Loss/macl, Loss/rpn_cls, Loss/cls
- Grad/norm
- Parameters (via ParameterMonitorHook)
- t-SNE embeddings (via TSNEVisualHook)

### Plotly 交互式报告
```bash
# 自动生成: metrics_logs/run_*/metrics_report.html
# 双击打开浏览器查看
```

**交互功能**:
- 缩放、平移
- 图例开关
- 数据点悬浮显示
- 导出 PNG

---

## 🎓 最佳实践

### 1. 训练前必做
✓ 运行 `verify_single_class_pipeline.py`  
✓ 运行 `test_forward_backward.py`  
✓ 运行 `gen_dataset_report.py`  
✓ 检查所有测试通过

### 2. 训练中监控
✓ 启用 MetricsExportHook  
✓ 启用 TSNEVisualHook  
✓ 启用 CheckInvalidLossHook  
✓ 定期查看 TensorBoard

### 3. 训练后分析
✓ 分析日志曲线  
✓ 生成混淆矩阵  
✓ 评估鲁棒性  
✓ 对比实验结果

### 4. 调试问题
✓ 检查 test*.py 输出  
✓ 查看 CheckInvalidLossHook 日志  
✓ 分析梯度流 (test_forward_backward.py)  
✓ 检查数据配对 (gen_dataset_report.py)

---

## 📌 总结

本项目包含 **445+ 个测试/验证/监控文件**，覆盖：

- ✅ **环境验证**: 12 个快速测试脚本
- ✅ **训练监控**: 9 个功能完备的 Hooks
- ✅ **数据验证**: 3 个数据集工具 + 配对检查
- ✅ **模型测试**: 404+ 个单元测试
- ✅ **性能分析**: 12 个分析工具
- ✅ **可视化**: t-SNE、曲线图、HTML 报告
- ✅ **文档**: 4 个详细技术文档

所有工具已验证通过，可直接用于 **LLVIP → KAIST → M3FD** 三阶段训练！

---

**文档版本**: v1.0  
**最后更新**: 2025-11-05
