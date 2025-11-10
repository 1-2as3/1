"""Quick dataset probe for KAIST paired loading shape consistency"""
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmengine.config import Config
from mmdet.registry import DATASETS
import torch

cfg = Config.fromfile('configs/llvip/stage2_llvip_kaist_finetune_sanity.py')
print('[1] Building train dataset...')
ds = DATASETS.build(cfg.train_dataloader['dataset'])
print(f'[2] Dataset length: {len(ds)}')

# Sample first 4 items to check shapes
shapes = []
for i in range(min(4, len(ds))):
    try:
        sample = ds[i]
        inp = sample.get('inputs')
        if torch.is_tensor(inp):
            shapes.append(inp.shape)
            print(f'  Sample {i}: inputs shape={inp.shape}')
        else:
            print(f'  Sample {i}: inputs type={type(inp)}')
    except Exception as e:
        print(f'  Sample {i}: ERROR {e}')

unique_shapes = set(shapes)
print(f'[3] Unique shapes: {unique_shapes}')
if len(unique_shapes) > 1:
    print('[WARN] Variable shapes detected - batch collation will fail!')
else:
    print('[OK] All shapes consistent.')
