"""Stage2 KAIST fine-tune (no domain loss) + FreezeMonitorHook"""
from mmengine.config import Config

_base_ = ['stage2_kaist_domain_ft_nodomain.py']

custom_hooks = [
    dict(type='FreezeBackboneHook', bn_eval=True),
    dict(type='FreezeMonitorHook', check_interval=1, strict=True, print_detail=True)
]
