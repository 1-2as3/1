# 域对齐模块使用说明

## 概述
域对齐模块通过对抗训练实现域不变特征学习，帮助模型在多个数据集上获得更好的泛化性能。

## 核心组件

### 1. GradientReversalLayer (梯度反转层)
- 前向传播：恒等映射
- 反向传播：反转并缩放梯度
- 作用：使特征提取器学习域不变特征

### 2. DomainClassifier (域分类器)
- 尝试区分不同数据集的特征
- 结合 GRL 实现对抗训练
- 结构：256 → 128 → num_domains

### 3. DomainAdaptationHook
- 动态调整 lambda 参数
- 支持线性和指数调度
- 自动记录训练日志

## 使用方法

### 阶段一：单数据集训练（不使用域对齐）
```python
# configs/llvip/faster_rcnn_r50_fpn_macl_msp_dhn.py
model = dict(
    roi_head=dict(
        use_domain_alignment=False,  # 不使用域对齐
        macl_head=dict(...)
    )
)
```

### 阶段二/三：多数据集训练（使用域对齐）
```python
# configs/llvip/faster_rcnn_r50_fpn_macl_msp_dhn_stage2_domain.py
model = dict(
    roi_head=dict(
        use_domain_alignment=True,
        domain_classifier=dict(
            type='DomainClassifier',
            in_dim=256,
            num_domains=3,  # LLVIP, KAIST, M3FD
            hidden_dim=128
        ),
        domain_loss_weight=0.1,
        macl_head=dict(...)
    )
)

# 数据集配置需要添加 domain_id
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='LLVIPDataset',
                metainfo=dict(domain_id=0, domain_name='LLVIP'),
                ...
            ),
            dict(
                type='KAISTDataset',
                metainfo=dict(domain_id=1, domain_name='KAIST'),
                ...
            ),
            dict(
                type='M3FDDataset',
                metainfo=dict(domain_id=2, domain_name='M3FD'),
                ...
            )
        ]
    )
)

# 添加域适应 Hook
custom_hooks = [
    dict(
        type='DomainAdaptationHook',
        initial_lambda=0.0,
        final_lambda=1.0,
        schedule='exp'
    )
]
```

## ROIHead 集成示例

在 ROIHead 的 loss() 方法中添加域损失：

```python
def loss(self, x, rpn_results_list, batch_data_samples):
    # ... 原有的检测损失 ...
    
    # 域对齐损失
    if self.use_domain_alignment:
        # 提取全局特征
        global_feat = self.extract_global_feat(x)
        
        # 获取当前 lambda 值
        lambda_p = getattr(self, 'domain_lambda', 1.0)
        
        # 域分类
        dom_pred = self.domain_classifier(global_feat, lambda_=lambda_p)
        
        # 获取域标签
        dom_labels = torch.tensor(
            [sample.metainfo.get('domain_id', 0) 
             for sample in batch_data_samples]
        ).to(dom_pred.device)
        
        # 计算域损失
        loss_domain = F.cross_entropy(dom_pred, dom_labels)
        losses.update({'loss_domain': self.domain_loss_weight * loss_domain})
    
    return losses
```

## Lambda 调度策略

### 指数调度 (推荐)
```python
lambda_p = 2 / (1 + exp(-10 * progress)) - 1
```
- 初期：lambda ≈ 0，专注于任务损失
- 中期：lambda 快速增长，开始域对齐
- 后期：lambda → 1，充分域对齐

### 线性调度
```python
lambda_p = progress  # 从 0 线性增长到 1
```
- 简单直观
- 适合调试和初步实验

## 超参数调优建议

1. **domain_loss_weight** (0.05 - 0.2)
   - 太小：域对齐效果不明显
   - 太大：可能影响检测性能

2. **lambda 调度**
   - initial_lambda: 通常设为 0.0
   - final_lambda: 0.5 - 1.0
   - schedule: 'exp' 效果通常优于 'linear'

3. **域分类器学习率**
   - 建议设为主网络的 10 倍
   - 帮助域分类器更快收敛

## 监控指标

训练时关注以下指标：
1. `loss_domain`: 域分类损失（应该保持在较高水平）
2. `domain_lambda`: 当前 lambda 值
3. `loss_macl`: 对比学习损失
4. `loss_rpn_cls`, `loss_bbox`: 检测损失

理想情况：
- loss_domain 保持高位（说明特征是域不变的）
- 检测损失正常下降
- 各数据集上的 mAP 都有提升

## 故障排除

### 问题 1：loss_domain 快速降至接近 0
**原因**：域分类器过强，特征提取器无法对抗
**解决**：
- 降低 domain_loss_weight
- 降低域分类器学习率
- 减慢 lambda 增长速度

### 问题 2：检测性能下降
**原因**：域对齐过强，损害了任务相关特征
**解决**：
- 降低 domain_loss_weight
- 延迟域对齐的开始时间
- 使用更温和的 lambda 调度

### 问题 3：不同数据集性能不平衡
**原因**：数据集大小差异或难度不同
**解决**：
- 使用 BalancedModalitySampler
- 为小数据集增加采样权重
- 调整各数据集的损失权重
