# ✅ 训练前检查清单（Stage2）

## 🔴 严重缺陷（必须修复）

- [ ] **Backbone 真正冻结**
  ```bash
  # 检查方法
  python -c "
  from mmengine.config import Config
  from mmdet.registry import MODELS
  from mmdet.utils import register_all_modules
  register_all_modules()
  cfg = Config.fromfile('configs/llvip/stage2_kaist_domain_ft.py')
  # ... 构建模型
  bb_trainable = sum(p.numel() for n,p in model.named_parameters() if 'backbone' in n and p.requires_grad)
  print(f'Backbone可训练参数: {bb_trainable:,}')
  assert bb_trainable == 0, '❌ Backbone未完全冻结!'
  "
  ```
  - ✗ 当前状态：23,282,688 个可训练
  - ✓ 目标状态：0 个可训练
  - 📝 修复方案：使用 `train_stage2_frozen.py`

- [ ] **Stage1 预训练权重存在**
  ```bash
  ls -lh work_dirs/stage1_llvip_pretrain/epoch_latest.pth
  ```
  - ✗ 当前状态：文件不存在
  - ✓ 解决方案 A：训练 Stage1
  - ✓ 解决方案 B：使用官方权重（临时）

- [ ] **推理元数据完整**
  ```bash
  python -c "
  from mmdet.datasets import KAISTDataset
  cfg = {'type':'KAISTDataset', 'data_root':'C:/KAIST_processed/', 'ann_file':'C:/KAIST_processed/ImageSets/test.txt', 'return_modality_pair':False}
  ds = KAISTDataset(**cfg)
  sample = ds[0]
  meta = sample['data_samples'].metainfo
  assert 'scale_factor' in meta, '❌ 缺少 scale_factor'
  print('✓ 元数据完整')
  "
  ```

## 🟡 潜在问题（建议检查）

- [ ] **MSP 模块参数可训练**
  ```bash
  # 预期：515 个参数，包括 alpha
  python -c "
  # ... 构建模型
  msp_params = [(n,p.numel()) for n,p in model.named_parameters() if 'msp' in n.lower()]
  print(f'MSP参数: {sum(p[1] for p in msp_params)}')
  assert len(msp_params) > 0, '❌ MSP模块未实例化'
  "
  ```

- [ ] **MACL 模块参数可训练**
  ```bash
  # 预期：49,729 个参数，包括 tau
  python -c "
  # ... 构建模型
  macl_params = [(n,p.numel()) for n,p in model.named_parameters() if 'macl' in n.lower()]
  print(f'MACL参数: {sum(p[1] for p in macl_params)}')
  assert len(macl_params) > 0, '❌ MACL模块未实例化'
  "
  ```

- [ ] **MACL 损失在训练中产生**
  ```bash
  # 训练5个epoch后检查
  grep "loss_macl" work_dirs/person_only_stage2/*/log.txt
  # 如果为空，说明MACL未激活
  ```

- [ ] **数据集路径存在**
  ```bash
  ls C:/KAIST_processed/ImageSets/{train,val,test}.txt
  ```

## 📊 参数统计验证

运行深度检查并核对数字：
```bash
python deep_check.py | tee deep_check_result.txt
```

预期结果：
```
总参数: 41.40M
├─ Backbone: 23.51M (冻结: 23.51M ✓, 可训练: 0 ✓)
├─ Neck: 3.34M (可训练)
├─ RPN: 0.59M (可训练)
├─ RoI Head: 13.90M (可训练)
├─ MSP: 0.0005M (可训练) ✓
└─ MACL: 0.05M (可训练) ✓

关键参数：
✓ neck.msp_module.alpha
✓ roi_head.macl_head.tau
✓ roi_head.macl_head.proj.0.weight
✓ roi_head.macl_head.proj.3.weight
```

## 🚀 训练启动验证

### 方案 A：使用修复版训练脚本（推荐）
```bash
python train_stage2_frozen.py
```

启动后应该看到：
```
[4/5] 冻结 Backbone 参数...
   已冻结 156 个 Backbone 参数

[5/5] 验证冻结状态...
   Backbone:
     总参数: 23,508,032 (23.51M)
     可训练: 0 (0.00M)  ✓
     冻结:   23,508,032 (23.51M)  ✓

✓ Backbone 已完全冻结
```

### 方案 B：标准训练脚本（需手动修改）
```bash
# 在 tools/train.py 中添加
# ... runner = Runner.from_cfg(cfg)
for name, param in runner.model.named_parameters():
    if 'backbone' in name:
        param.requires_grad = False
# runner.train()

python tools/train.py configs/llvip/stage2_kaist_domain_ft.py
```

## 📝 训练中监控

### 1. 日志实时查看
```bash
tail -f work_dirs/person_only_stage2/*/log.txt
```

关键指标：
- `loss_rpn_cls`, `loss_rpn_bbox` - RPN 损失
- `loss_cls`, `loss_bbox` - RoI 损失
- `loss_macl` - MACL 损失（如果有）
- `lr` - 学习率变化

### 2. Backbone 冻结验证（训练中）
```bash
# 每个epoch后检查
python -c "
import torch
ckpt = torch.load('work_dirs/person_only_stage2/epoch_1.pth')
# 检查优化器状态中是否有 backbone 参数
bb_in_opt = [k for k in ckpt.get('optimizer',{}).get('state',{}).keys() if 'backbone' in str(k)]
print(f'优化器中 Backbone 参数数: {len(bb_in_opt)}')
if len(bb_in_opt) > 0:
    print('❌ 警告：Backbone 参数在优化器中（可能未冻结）')
else:
    print('✓ Backbone 正确冻结')
"
```

### 3. 显存使用监控
```bash
nvidia-smi -l 1
```

预期：
- 训练显存：~2-4GB（batch_size=2）
- 如果 >6GB，可能 Backbone 未冻结

## 🎯 验证通过标准

- [x] 所有 🔴 严重缺陷已修复
- [x] Backbone 完全冻结（0个可训练参数）
- [x] MSP 和 MACL 模块存在且可训练
- [x] 深度检查无错误
- [x] 前向传播测试通过
- [ ] 训练5个epoch无异常
- [ ] 验证集 mAP 正常（>30%）

## ⚠️  常见问题

### Q1: "Backbone 有可训练参数"怎么办？
A: 使用 `train_stage2_frozen.py` 而不是标准训练脚本

### Q2: "未检测到 loss_macl"怎么办？
A: 
1. 确认 `return_modality_pair=True`（如需配对训练）
2. 检查 RoI Head 是否正确调用 MACL
3. 暂时可以先单模态训练，后续改进

### Q3: "Stage1 权重不存在"怎么办？
A:
- **推荐**：训练 Stage1（1-2天）
- **临时**：使用官方权重（性能略降）
- **测试**：从头训练（仅概念验证）

### Q4: 如何确认训练正常？
A: 查看以下指标
- Epoch 1: loss ~1.5-2.5
- Epoch 5: loss ~0.8-1.2
- Epoch 12: loss ~0.5-0.8
- Val mAP: 逐步上升

---

**最后提醒**：
修复这些缺陷后，请重新运行 `deep_check.py` 确认所有问题已解决！

```bash
python deep_check.py && echo "✓ 检查通过，可以开始训练"
```
