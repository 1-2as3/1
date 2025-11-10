"""Stage2 KAIST Finetune (sanity 3 epochs, FP16, spawn on Windows)

This derives from stage2_llvip_kaist_finetune.py but:
- sets env_cfg.mp_cfg.mp_start_method = 'spawn' for Windows
- sets num_workers=0 and persistent_workers=False for all dataloaders
- reduces max_epochs to 3 and uses a dedicated work_dir
"""

from mmengine.config import read_base

with read_base():
    from .stage2_llvip_kaist_finetune import *  # noqa: F401,F403

# Windows-safe multiprocessing
env_cfg = dict(
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
    cudnn_benchmark=False,
)

# Dataloaders: safer workers for sanity on Windows
train_dataloader['num_workers'] = 0  # noqa: F821
train_dataloader['persistent_workers'] = False  # noqa: F821
val_dataloader['num_workers'] = 0  # noqa: F821
val_dataloader['persistent_workers'] = False  # noqa: F821
test_dataloader['num_workers'] = 0  # noqa: F821
test_dataloader['persistent_workers'] = False  # noqa: F821

# Short run
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)

# Lighter TSNE
custom_hooks[0]['num_samples'] = 150  # noqa: F821

work_dir = './work_dirs/stage2_kaist_finetune_sanity'

# 修正 best checkpoint 指标名称：VOCMetric 带前缀 'pascal_voc'
default_hooks['checkpoint']['save_best'] = 'pascal_voc/mAP'  # noqa: F821
