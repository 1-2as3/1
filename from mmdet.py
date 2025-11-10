from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
import mmdet.datasets.transforms.formatting as fmt
from mmengine.registry import TRANSFORMS
print('Has PackDetInputs attr?', hasattr(fmt,'PackDetInputs'))
cls=getattr(fmt,'PackDetInputs')
from inspect import isclass
print('Is class', isclass(cls))
try:
    TRANSFORMS.register_module()(cls)
except Exception as e:
    print('register error', e)
print('Size after manual register', len(TRANSFORMS.module_dict))
print('PackDetInputs now?', 'PackDetInputs' in TRANSFORMS.module_dict)