# Quick Fix for Registry Error on Remote Server

## Problem
```
KeyError: 'PairedDetDataPreprocessor is not in the mmdet::model registry'
```

## Root Cause
远程服务器缺少自定义代码文件，或 Python 未正确导入自定义模块。

---

## Solution 1: 同步代码到远程（推荐）

### 使用自动同步脚本
```bash
# 在本地执行（Windows 需要 WSL 或 Git Bash）
chmod +x sync_custom_code.sh
bash sync_custom_code.sh user@remote-server:~/mmdetection_remote

# 示例
bash sync_custom_code.sh user@192.168.1.100:~/mmdetection_remote
```

### 手动同步（若脚本失败）
```bash
# 同步自定义模块
scp -r mmdet/models/data_preprocessors/paired_preprocessor.py \
       user@remote:~/mmdetection_remote/mmdet/models/data_preprocessors/

scp -r mmdet/models/macldhnmsp/ \
       user@remote:~/mmdetection_remote/mmdet/models/

scp mmdet/models/roi_heads/aligned_roi_head.py \
    user@remote:~/mmdetection_remote/mmdet/models/roi_heads/

scp mmdet/models/utils/domain_aligner.py \
    user@remote:~/mmdetection_remote/mmdet/models/utils/

scp mmdet/engine/hooks/domain_weight_warmup_hook.py \
    user@remote:~/mmdetection_remote/mmdet/engine/hooks/

scp mmdet/datasets/kaist.py \
    user@remote:~/mmdetection_remote/mmdet/datasets/

# 同步 __init__.py 文件
scp mmdet/models/data_preprocessors/__init__.py \
    user@remote:~/mmdetection_remote/mmdet/models/data_preprocessors/

scp mmdet/models/__init__.py \
    user@remote:~/mmdetection_remote/mmdet/models/

scp mmdet/datasets/__init__.py \
    user@remote:~/mmdetection_remote/mmdet/datasets/

# 同步配置文件
scp -r configs/llvip/ \
       user@remote:~/mmdetection_remote/configs/
```

---

## Solution 2: 远程验证与修复

### SSH 到远程服务器
```bash
ssh user@remote-server
cd ~/mmdetection_remote
conda activate mmdet_py311
```

### 验证文件存在
```bash
# 检查关键文件
ls -lh mmdet/models/data_preprocessors/paired_preprocessor.py
ls -lh mmdet/models/macldhnmsp/__init__.py
ls -lh mmdet/models/roi_heads/aligned_roi_head.py
ls -lh configs/llvip/stage2_kaist_full_conservative.py
```

### 测试导入
```bash
python -c "
from mmdet.models.data_preprocessors import PairedDetDataPreprocessor
print('✓ PairedDetDataPreprocessor imported')

from mmdet.models.roi_heads import AlignedRoIHead
print('✓ AlignedRoIHead imported')

from mmdet.models.macldhnmsp import MACLHead
print('✓ MACLHead imported')

from mmdet.datasets import KAISTDataset
print('✓ KAISTDataset imported')

from mmdet.engine.hooks import DomainWeightWarmupHook
print('✓ DomainWeightWarmupHook imported')
"
```

### 检查注册表
```bash
python -c "
from mmdet.registry import MODELS, DATASETS, HOOKS

print('Checking MODELS registry:')
if 'PairedDetDataPreprocessor' in MODELS.module_dict:
    print('  ✓ PairedDetDataPreprocessor')
else:
    print('  ✗ PairedDetDataPreprocessor NOT FOUND')

if 'AlignedRoIHead' in MODELS.module_dict:
    print('  ✓ AlignedRoIHead')
else:
    print('  ✗ AlignedRoIHead NOT FOUND')

print('\\nChecking DATASETS registry:')
if 'KAISTDataset' in DATASETS.module_dict:
    print('  ✓ KAISTDataset')
else:
    print('  ✗ KAISTDataset NOT FOUND')

print('\\nChecking HOOKS registry:')
if 'DomainWeightWarmupHook' in HOOKS.module_dict:
    print('  ✓ DomainWeightWarmupHook')
else:
    print('  ✗ DomainWeightWarmupHook NOT FOUND')
"
```

---

## Solution 3: 配置文件已更新（custom_imports）

最新的配置文件已添加 `custom_imports` 显式声明：
```python
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.paired_preprocessor',
        'mmdet.models.macldhnmsp',
        'mmdet.models.roi_heads.aligned_roi_head',
        'mmdet.models.utils.domain_aligner',
        'mmdet.engine.hooks.domain_weight_warmup_hook',
        'mmdet.datasets.kaist'
    ],
    allow_failed_imports=False
)
```

确保远程的配置文件也包含此段（重新同步 configs/llvip/）。

---

## Solution 4: 重新安装 mmdet（若改动未生效）

```bash
cd ~/mmdetection_remote
pip uninstall mmdet -y
pip install -e . -v
```

---

## 完整修复流程（推荐顺序）

### Step 1: 本地同步代码
```bash
# 在本地 Windows (使用 Git Bash 或 WSL)
cd C:/Users/Xinyu/mmdetection
bash sync_custom_code.sh user@remote:~/mmdetection_remote
```

### Step 2: 远程验证
```bash
# SSH 到远程
ssh user@remote
cd ~/mmdetection_remote
conda activate mmdet_py311

# 快速验证
python -c "from mmdet.models.data_preprocessors import PairedDetDataPreprocessor; print('OK')"
```

### Step 3: 重启训练
```bash
python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \
    --work-dir work_dirs/stage2_kaist_full_conservative_remote
```

---

## 常见错误与解决

### Error: "ModuleNotFoundError: No module named 'mmdet.models.macldhnmsp'"
```bash
# 检查目录是否存在
ls mmdet/models/macldhnmsp/
# 若不存在，重新同步
scp -r mmdet/models/macldhnmsp/ user@remote:~/mmdetection_remote/mmdet/models/
```

### Error: "ImportError: cannot import name 'AlignedRoIHead'"
```bash
# 检查 __init__.py 是否更新
cat mmdet/models/__init__.py | grep AlignedRoIHead
# 若无，重新同步
scp mmdet/models/__init__.py user@remote:~/mmdetection_remote/mmdet/models/
```

### Error: Config 仍报 KeyError
```bash
# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 重新运行
python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \
    --work-dir work_dirs/stage2_kaist_full_conservative_remote
```

---

## 快速诊断命令

```bash
# 一键诊断脚本（远程执行）
python - <<'PY'
import sys
errors = []

try:
    from mmdet.models.data_preprocessors import PairedDetDataPreprocessor
    print('✓ PairedDetDataPreprocessor')
except Exception as e:
    errors.append(f'✗ PairedDetDataPreprocessor: {e}')

try:
    from mmdet.models.roi_heads import AlignedRoIHead
    print('✓ AlignedRoIHead')
except Exception as e:
    errors.append(f'✗ AlignedRoIHead: {e}')

try:
    from mmdet.models.macldhnmsp import MACLHead
    print('✓ MACLHead')
except Exception as e:
    errors.append(f'✗ MACLHead: {e}')

try:
    from mmdet.datasets import KAISTDataset
    print('✓ KAISTDataset')
except Exception as e:
    errors.append(f'✗ KAISTDataset: {e}')

try:
    from mmdet.engine.hooks import DomainWeightWarmupHook
    print('✓ DomainWeightWarmupHook')
except Exception as e:
    errors.append(f'✗ DomainWeightWarmupHook: {e}')

if errors:
    print('\n=== ERRORS ===')
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print('\n✓ All custom modules OK')
PY
```

---

## 联系与日志

若问题持续，保存以下信息：
```bash
# 生成诊断报告
{
    echo "=== File Check ==="
    find mmdet/models -name "*.py" | grep -E 'paired|macl|aligned|domain'
    
    echo -e "\n=== Import Test ==="
    python -c "from mmdet.models.data_preprocessors import PairedDetDataPreprocessor" 2>&1
    
    echo -e "\n=== Registry ==="
    python -c "from mmdet.registry import MODELS; print([k for k in MODELS.module_dict.keys() if 'Paired' in k or 'Aligned' in k])"
    
    echo -e "\n=== Python Path ==="
    python -c "import sys; print('\n'.join(sys.path))"
} > registry_debug.txt

cat registry_debug.txt
```
