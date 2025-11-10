"""Stage2 KAIST fine-tune (no domain loss) + Freeze hooks (lazy-import friendly)
This wraps the lazy base config and appends freeze-related hooks, avoiding mixed lazy/non-lazy chain.
"""
from mmengine.config import read_base

with read_base():
    from .stage2_kaist_domain_ft_nodomain import *  # noqa: F401,F403

# Append/override hooks in lazy style
custom_hooks = [
    dict(type='FreezeBackboneHook', bn_eval=True),
    dict(type='FreezeMonitorHook', check_interval=1, strict=True, print_detail=True)
]
