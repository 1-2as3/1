import os, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.chdir(str(root))
if __name__ == '__main__':
    with open(root / 'test_stage3_config.py', 'rb') as f:
        code = compile(f.read(), str(root / 'test_stage3_config.py'), 'exec')
        exec(code, {})
