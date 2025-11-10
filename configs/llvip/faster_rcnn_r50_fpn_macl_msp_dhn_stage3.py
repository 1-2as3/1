from mmengine.config import read_base

with read_base():
    from .faster_rcnn_r50_fpn_macl_msp_dhn import *

# 继承所有基础配置，只修改数据加载部分
train_dataloader.update(
    dict(
        batch_size=2,
        num_workers=2,
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='KAISTDataset',
                    data_root='path/to/KAIST',  # 替换为实际的 KAIST 数据集路径
                    ann_file='path/to/KAIST/annotations/train.txt',
                    data_prefix=dict(
                        visible='path/to/KAIST/visible/train',
                        infrared='path/to/KAIST/lwir/train'
                    )
                ),
                dict(
                    type='M3FDDataset',
                    data_root='path/to/M3FD',  # 替换为实际的 M3FD 数据集路径
                    ann_file='path/to/M3FD/annotations/train.txt',
                    data_prefix=dict(
                        visible='path/to/M3FD/visible/train',
                        infrared='path/to/M3FD/infrared/train'
                    )
                )
            ]
        ),
        sampler=dict(type='BalancedModalitySampler')
    )
)

# Person-only detection across all datasets (LLVIP, KAIST, M3FD)
model.roi_head.bbox_head.num_classes = 1

# 其他配置保持不变