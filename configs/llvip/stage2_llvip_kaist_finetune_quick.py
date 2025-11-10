"""Stage2 KAIST Quick Subset Config (首50样本快速验证)

派生自 stage2_llvip_kaist_finetune_sanity.py，进一步限制数据量：
- 仅加载前50个样本（利用 MMEngine 的 indices 参数）
- max_epochs 保持3
- 确保快速通过 dataset 构建和第一次 val

用途：
1. 快速定位卡死问题（若子集不卡，即可排除代码逻辑错误）
2. 端到端验证 pipeline/loss/hook 工作是否正常
"""

from mmengine.config import read_base

with read_base():
    from .stage2_llvip_kaist_finetune_sanity import *  # noqa: F401,F403

# 🔑 关键修改1: 限制数据集为前50个样本
train_dataloader['dataset']['indices'] = list(range(50))  # noqa: F821
val_dataloader['dataset']['indices'] = list(range(20))  # noqa: F821
test_dataloader['dataset']['indices'] = list(range(20))  # noqa: F821

# 🔑 关键修改2: 禁用 return_modality_pair 避免全量 data_list 遍历（_get_paired_data 内部）
# 该模式会在 __getitem__ 时 for-loop self.data_list（即使 indices 有限制，data_list 仍是全量解析后的）
train_dataloader['dataset']['return_modality_pair'] = False  # noqa: F821
val_dataloader['dataset']['return_modality_pair'] = False  # noqa: F821
test_dataloader['dataset']['return_modality_pair'] = False  # noqa: F821

# 针对小数据集调整验证间隔和日志间隔
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
default_hooks['logger']['interval'] = 10  # noqa: F821

work_dir = './work_dirs/stage2_kaist_quick'
