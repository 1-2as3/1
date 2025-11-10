"""Stage3 joint multimodal training + FreezeMonitorHook"""
_base_ = ['stage3_joint_multimodal.py']

custom_hooks = [
    dict(type='FreezeMonitorHook', check_interval=1, strict=False, print_detail=True)
]
