# 训练全流程说明（LLVIP → KAIST → Stage3）

本指南覆盖从数据准备、环境验证、训练监控到结果导出与评估的完整流程。

## 1. 数据准备

- 路径约定（Windows）
  - LLVIP: C:/LLVIP/LLVIP
  - KAIST: C:/KAIST_processed
  - M3FD: C:/M3FD
- 结构要求（KAIST）
  - C:/KAIST_processed/
    - Annotations/  # VOC XML
    - visible/
    - infrared/
    - ImageSets/train.txt, val.txt, test.txt

可选：生成数据统计报告
```bash
python tools/gen_dataset_report.py --dataset KAIST --data-root C:/KAIST_processed --output-dir analysis_report
```

## 2. 快速验证与分步排查

- 一键验证（推荐）
```bash
python verify_all.py
```
- 分步验证（构建/数据/前向）
```bash
python test_stage2_build.py
python test_dataset_kaist.py
python test_forward_kaist.py
```
- 可视化/开关/Stage3 检查（可选）
```bash
python test_kaist_visualization.py
python test_module_switches.py
python test_stage3_config.py
```

## 3. 开始训练

- Stage 1（LLVIP 预训练）
```bash
python tools/train.py configs/llvip/stage1_llvip_pretrain.py --amp
```
- Stage 2（KAIST 微调/域适应）
```bash
python tools/train.py configs/llvip/stage2_kaist_domain_ft.py --amp
```
- Stage 3（联合训练/增强）
```bash
python tools/train.py configs/llvip/stage3_joint_multimodal.py --amp
```
常用参数：
- --work-dir <dir> 自定义输出目录
- --resume 从最新断点恢复
- --amp 启用混合精度

## 4. 训练监控（建议开启）

在相应 config 中启用：
```python
default_hooks = dict(
    metrics_export=dict(type='MetricsExportHook', interval=1),
    tsne_visual=dict(type='TSNEVisualHook', interval=1),
    checkloss=dict(type='CheckInvalidLossHook', interval=50)
)
```
查看：
- TensorBoard: work_dirs/<exp>/tensorboard_logs
- 指标导出：work_dirs/metrics_logs/run_*/（csv/png/html/zip）

## 5. 结果位置与导出

- Checkpoints: work_dirs/<exp>/epoch_*.pth, latest.pth
- 训练日志：work_dirs/<exp>/*.log.json
- 指标汇总：work_dirs/metrics_logs/run_*/metrics.csv
- HTML 报告：work_dirs/metrics_logs/run_*/metrics_report.html
- t-SNE 可视化：work_dirs/tsne_vis/*.png

导出推理结果（示例）：
```bash
python tools/test.py <config> <checkpoint> --out results.pkl --show-dir outputs
```

## 6. 评估与分析

- 日志曲线：
```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/*/log.json --keys loss_total
```
- 混淆矩阵：
```bash
python tools/analysis_tools/confusion_matrix.py <config> results.pkl --show --out confusion.png
```
- FLOPs：
```bash
python tools/analysis_tools/get_flops.py <config> --shape 640 512
```

## 7. 常见问题（Quick Fix）

- 构建失败（缺少 backbone/roi_head）：
  - 运行 test_stage2_build.py，触发“基础模型合并回退”逻辑
- 前向 dtype 错误（Byte→Float）：
  - 确保通过 model.data_preprocessor 预处理输入
- num_classes 不匹配：
  - 确保 roi_head.bbox_head.num_classes=1
- 数据加载失败：
  - 检查 data_root 与 ImageSets 是否正确，先运行 test_dataset_kaist.py

## 8. 附：脚本与文档索引

- 脚本：support/scripts/
- 文档：support/docs/（本文件夹）
- KAIST 报告：reports/KAIST_INTEGRATION_REPORT.md

完成以上步骤后，即可获得稳定可复现的训练与评估流程。祝实验顺利！
