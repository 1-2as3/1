"""
MAN Module Integration Analysis and Implementation Guide

当前状态：
1. ModalityAdaptiveNorm 已存在于 mmdet/models/macldhnmsp/modality_adaptive_norm.py
2. 但未集成到 StandardRoIHead 中
3. 用户建议的 MANAligner 是不同概念（跨模态对齐 vs 模态自适应归一化）

问题分析：
============

原建议的致命缺陷：
------------------

1. **概念混淆**
   - 原建议: MANAligner 实现跨模态对齐（MSE loss between vis/ir features）
   - 现有实现: ModalityAdaptiveNorm 实现模态自适应 BN
   - 问题: 两者是不同的模块，名称相似但功能完全不同

2. **覆盖风险**
   - 原建议: 创建 mmdet/models/utils/man_aligner.py
   - 问题: 会与现有 modality_adaptive_norm.py 造成概念混乱
   - 后果: 代码维护困难，功能重复

3. **append 模式错误**
   - 原建议: 用 open(stage3_cfg, 'a') 追加配置
   - 问题: 会在文件末尾追加，导致语法错误（在文件结束后添加内容）
   - 后果: 配置文件损坏，无法加载

4. **缺少集成点**
   - 原建议只创建模块，没有说明如何在 StandardRoIHead 中调用
   - 缺少 forward 流程集成
   - 缺少损失计算集成

5. **未处理现有 lambda 权重**
   - Stage3 已有 lambda1, lambda2, lambda3
   - 原建议未说明新的 align_loss_weight 如何与现有权重协调
   - 可能导致损失权重冲突

正确的实现方案：
================

方案一：激活现有的 ModalityAdaptiveNorm（推荐）
--------------------------------------------
优点：
- 模块已实现且已注册
- 无需修改代码，只需配置
- 在 backbone 层面进行模态自适应

实现步骤：
1. 修改 Stage3 配置，使用 ModalityAdaptiveResNet 作为 backbone
2. 在 data_preprocessor 中确保传递 modality 元信息
3. 无需修改 StandardRoIHead

配置示例：
```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ModalityAdaptiveResNet',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            # ... 其他参数
        ),
        use_modality_bn=True,
        modality_bn_layers=[0, 1]  # 在前两层使用模态自适应 BN
    ),
    # ... 其余配置
)
```

方案二：实现跨模态对齐模块（如果确实需要）
-----------------------------------------
如果您真正需要的是跨模态特征对齐（而非模态自适应 BN），应该：

1. 创建新模块，避免与 MAN 混淆
   - 命名: CrossModalAligner 或 ModalityAlignmentHead
   - 位置: mmdet/models/macldhnmsp/cross_modal_aligner.py

2. 集成到 StandardRoIHead
   - 添加 use_cross_modal_align 参数
   - 在 loss() 方法中计算对齐损失
   - 添加 lambda4 权重

3. 修改配置（不使用 append）
   - 在 roi_head dict 中添加配置
   - 不破坏现有结构

实现示例：
```python
# mmdet/models/macldhnmsp/cross_modal_aligner.py
@MODELS.register_module()
class CrossModalAligner(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128):
        super().__init__()
        self.vis_proj = nn.Linear(in_dim, hidden_dim)
        self.ir_proj = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, vis_feat, ir_feat):
        # 实现跨模态对齐
        pass
```

配置修改（正确方式）：
```python
# 在 Stage3 配置文件中直接修改 model dict
model = dict(
    # ... 现有配置
    roi_head=dict(
        type='StandardRoIHead',
        # ... 现有参数
        use_cross_modal_align=True,
        cross_modal_aligner=dict(
            type='CrossModalAligner',
            in_dim=256,
            hidden_dim=128
        ),
        lambda4=0.2,  # 跨模态对齐损失权重
    )
)
```

结论：
======

原建议**不周全**，存在严重问题：
1. ❌ 概念混淆（MANAligner vs ModalityAdaptiveNorm）
2. ❌ 文件覆盖风险
3. ❌ append 模式破坏配置文件
4. ❌ 缺少集成代码
5. ❌ 权重冲突未处理

推荐方案：
1. 如需模态自适应 BN：使用方案一（激活现有 ModalityAdaptiveResNet）
2. 如需跨模态对齐：使用方案二（创建 CrossModalAligner + 完整集成）

建议先明确需求：
- 需要模态自适应归一化（不同模态用不同 BN）？→ 方案一
- 需要跨模态特征对齐（让 vis/ir 特征相似）？→ 方案二
"""

# 保存分析报告
with open('reports/MAN_INTEGRATION_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.write(__doc__)

print("Analysis saved to: reports/MAN_INTEGRATION_ANALYSIS.md")
