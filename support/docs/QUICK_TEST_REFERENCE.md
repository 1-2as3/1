# 测试/验证代码快速参考表
**项目**: MMDetection 3.x + 多模态目标检测  
**生成**: 2025-11-05

---

## 🚀 快速启动检查清单

### ✅ 训练前必做（1 步即可）

```bash
python verify_all.py
```

说明：非交互式一键验证（3 秒倒计时后自动开始），依次完成“配置与模型构建 → 数据加载 → 前向/损失/反向”。

如需分步排查，可按下面“核心测试脚本速查”逐个执行。

---

## 📋 核心测试脚本速查

| 用途 | 脚本 | 耗时 | 关键输出 |
|------|------|------|---------|
| **一键验证（推荐）** | `verify_all.py` | 30-90s | 构建/数据/前向三项结果 + 故障排查提示 |
| **配置与模型构建** | `test_stage2_build.py` | 10-20s | 深度合并 base 的回退构建；num_classes=1 校验 |
| **数据加载与管道** | `test_dataset_kaist.py` | 15-30s | 单样本/批处理检查；统计信息 |
| **前向/损失/反向** | `test_forward_kaist.py` | 15-30s | data_preprocessor 生效；loss/backward 成功 |
| 可视化（可选） | `test_kaist_visualization.py` | 10-30s | 若干样本可视化图保存 |
| 模块开关（可选） | `test_module_switches.py` | 5-15s | MACL/MSP/DHN 开关验证 |
| Stage3 配置（可选） | `test_stage3_config.py` | 5-10s | 学习率调度与损失权重覆盖项 |

---

## 🎯 监控 Hook 配置

### 基础配置 (必备)
```python
default_hooks = dict(
    checkloss=dict(type='CheckInvalidLossHook', interval=50)
)
```

### 标准配置 (推荐)
```python
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50),
    tsne_visual=dict(type='TSNEVisualHook', interval=1, num_samples=200)
)
```

### 完整配置 (研究用)
```python
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    tsne_visual=dict(type='TSNEVisualHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50),
    parameter_monitor=dict(type='ParameterMonitorHook', interval=1),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1)
)
```

---

## 📊 数据集工具速查

### LLVIP 数据检查
```bash
python tools/gen_dataset_report.py \
    --dataset LLVIP \
    --data-root C:/LLVIP/LLVIP \
    --output-dir analysis_report
```

### KAIST 数据检查
```bash
python tools/gen_dataset_report.py \
    --dataset KAIST \
    --data-root C:/KAIST \
    --split-ratio 0.8
```

### M3FD 数据检查
```bash
python tools/gen_dataset_report.py \
    --dataset M3FD \
    --data-root C:/M3FD
```

**输出**: `analysis_report/` 包含 PNG 图表 + JSON 统计

---

## 🔍 分析工具速查

### 训练日志分析
```bash
python tools/analysis_tools/analyze_logs.py \
    plot_curve \
    work_dirs/*/log.json \
    --keys loss_total loss_macl \
    --out loss_curves.png
```

### 模型复杂度
```bash
python tools/analysis_tools/get_flops.py \
    configs/llvip/stage1_llvip_pretrain.py \
    --shape 640 512
```

### 混淆矩阵
```bash
python tools/analysis_tools/confusion_matrix.py \
    configs/llvip/stage2_kaist_domain_ft.py \
    results.pkl \
    --show --out confusion.png
```

### TensorBoard 启动
```bash
tensorboard --logdir work_dirs/stage1_llvip_pretrain/tensorboard_logs
# 访问 http://localhost:6006
```

---

## 🧪 测试文件功能对照表

| 测试文件 | 主要测试内容 | 失败时检查 |
|---------|-------------|-----------|
| verify_all.py | 一键全链路验证 | 查看错误提示，按建议定位至单项脚本 |
| test_stage2_build.py | 配置与模型构建 | 基础配置合并是否缺失；roi_head/bbox_head 覆盖项 |
| test_dataset_kaist.py | 数据加载/管道 | 数据根路径；标注格式；pipeline 变换 |
| test_forward_kaist.py | 前向/损失/反向 | data_preprocessor 是否生效；dtype/归一化 |
| test_kaist_visualization.py | 可视化 | 输出目录权限；样本索引 |
| test_module_switches.py | 模块开关 | use_macl/use_msp/use_dhn 配置项 |
| test_stage3_config.py | Stage3 配置 | T_max/eta_min 与 loss 权重是否覆盖成功 |

---

## 📈 监控输出位置

| 监控内容 | 输出位置 | 格式 |
|---------|---------|------|
| **训练指标** | `work_dirs/metrics_logs/run_*/` | CSV, PNG, HTML |
| **t-SNE 可视化** | `work_dirs/tsne_vis/` | PNG |
| **TensorBoard** | `work_dirs/*/tensorboard_logs/` | Events |
| **配置快照** | `work_dirs/metrics_logs/run_*/` | .py |
| **压缩备份** | `work_dirs/metrics_logs/` | .zip |
| **数据集报告** | `analysis_report/` | PNG, JSON |

---

## 🎯 常见问题速查

| 问题 | 检查脚本 | 解决方向 |
|------|---------|---------|
| 模型构建失败（缺 backbone/roi_head 等） | test_stage2_build.py | 使用脚本内“base 合并回退”逻辑；检查 read_base 继承 |
| 前向时报 dtype 错误（Byte→Float） | test_forward_kaist.py | 确保经 model.data_preprocessor；检查归一化与设备 |
| 数据加载失败/样本为空 | test_dataset_kaist.py | 检查数据根路径、标注、pipeline；打印首个样本 |
| num_classes 不匹配 | test_stage2_build.py | 覆盖 roi_head.bbox_head.num_classes=1 并验证 |
| 可视化无输出 | test_kaist_visualization.py | 检查保存目录与样本索引范围 |

---

## 🔧 调试流程

### 场景 1: 训练无法启动
```bash
1. python test1.py              # 检查环境
2. python test5.py              # 检查配置
3. python test6.py              # 检查 pipeline
4. 查看详细日志
```

### 场景 2: Loss 异常
```bash
1. python test_forward_backward.py  # 梯度稳定性
2. 检查 CheckInvalidLossHook 日志
3. 降低学习率重试
4. 检查数据归一化
```

### 场景 3: mAP 不收敛
```bash
1. python tools/analysis_tools/analyze_logs.py  # 分析曲线
2. python tools/analysis_tools/confusion_matrix.py  # 混淆矩阵
3. 检查 t-SNE 对齐 (TSNEVisualHook)
4. 调整损失权重
```

### 场景 4: 数据问题
```bash
1. python tools/gen_dataset_report.py  # 数据统计
2. python tools/analysis_tools/browse_dataset.py  # 可视化
3. 检查配对完整性
4. 验证标注格式
```

---

## 📚 文档索引

| 文档 | 内容 | 查看场景 |
|------|------|---------|
| `TEST_VERIFICATION_SUMMARY.md` | 完整测试总结 | 了解全部测试 |
| `PERSON_ONLY_MIGRATION.md` | Person-only 迁移 | 单类别配置 |
| `MODULE_SWITCHES_ENHANCEMENT_REPORT.md` | 模块开关 | 控制 MACL/MSP/DHN |
| `PAIRED_MODALITY_IMPLEMENTATION.md` | 成对模态 | 双流数据加载 |
| `ROBUST_LOSS_AGGREGATION_FIX.md` | 损失聚合 | 混合类型处理 |

---

## 💡 训练监控最佳实践

### 阶段 1: LLVIP 预训练
```python
# configs/llvip/stage1_llvip_pretrain.py
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50)
)
```
**监控重点**: loss_total, loss_macl 下降趋势

### 阶段 2: KAIST 域适应
```python
# configs/llvip/stage2_kaist_domain_ft.py
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    tsne_visual=dict(type='TSNEVisualHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50)
)
```
**监控重点**: t-SNE 对齐, loss_domain 收敛

### 阶段 3: 联合训练
```python
# configs/llvip/stage3_joint_multimodal.py
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    tsne_visual=dict(type='TSNEVisualHook', interval=1),
    parameter_monitor=dict(type='ParameterMonitorHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50)
)
```
**监控重点**: mAP 提升, 参数稳定性

---

## ⚡ 性能优化检查

| 检查项 | 工具 | 目标值 |
|-------|------|--------|
| FLOPs | get_flops.py | <100G |
| 参数量 | test1.py | <50M |
| 推理速度 | benchmark.py | >30 FPS |
| GPU 显存 | MemoryProfilerHook | <8GB |
| 梯度范数 | MetricsExportHook | 1-100 |
| 损失平衡 | test_forward_backward.py | 最大占比 <80% |

---

## 🎓 代码质量检查

```bash
# 代码格式
flake8 mmdet/

# 类型检查
mypy mmdet/

# 单元测试
pytest tests/ -v

# 覆盖率
pytest --cov=mmdet tests/
```

---

**快速参考**: 保存此文件，训练时随时查阅！  
**完整文档**: 参考 `TEST_VERIFICATION_SUMMARY.md`
