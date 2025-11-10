# 辅助脚本与说明文档索引（统一入口）

本目录集中收纳了项目的辅助脚本与关键说明文档，便于统一查阅与执行。

## 📂 脚本（support/scripts/）

- verify_all.py — 一键全链路验证（调用根目录同名脚本）
- test_stage2_build.py — 配置与模型构建（含 base 合并回退）
- test_dataset_kaist.py — KAIST 数据/管道/批处理检查
- test_forward_kaist.py — 前向/损失/反向（经 data_preprocessor）
- test_kaist_visualization.py — 小样本可视化
- test_module_switches.py — 模块开关验证
- test_stage3_config.py — Stage3 配置项检查

提示：以上脚本均为“包装器”，调用的是工程根目录下的同名脚本，确保路径/依赖一致。

## 📚 文档（support/docs/）

- TEST_VERIFICATION_SUMMARY.md — 完整测试/验证/监控总结
- QUICK_TEST_REFERENCE.md — 测试/验证快速参考表
- README_CONFIG_BUILD.md — 配置构建问题：完整解答（与 CONFIG_MERGE_EXPLAINED.md 配套）
- （参考）CONFIG_MERGE_EXPLAINED.md — 为什么测试脚本需要手动合并基础配置？（原文）
- PERSON_ONLY_MIGRATION.md — 单类（person-only）迁移总结
- MODULE_SWITCHES_ENHANCEMENT_REPORT.md — 模块开关增强报告
- PAIRED_MODALITY_IMPLEMENTATION.md — 成对模态实现报告
- ROBUST_LOSS_AGGREGATION_FIX.md — 鲁棒损失聚合修复
- STAGE3_CONFIG_SUMMARY.md — Stage3 配置概述
- TRAINING_GUIDE.md — 训练全流程说明（新增）

其他：
- KAIST_INTEGRATION_REPORT.md — 已迁移至 reports/ 目录（归档）

## 🚀 建议入口

1) 快速验证：support/scripts/verify_all.py（或根目录 verify_all.py）
2) 分步验证：support/scripts/test_* 系列
3) 文档查阅：support/docs/ 下相关主题
4) 完整流程：support/docs/TRAINING_GUIDE.md
