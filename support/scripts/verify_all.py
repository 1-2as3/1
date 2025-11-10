# Wrapper to call root-level verify_all.py; kept for unified access under support/scripts.
import os, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.chdir(str(root))
if __name__ == '__main__':
    # Execute the original script from the project root to preserve paths
    with open(root / 'verify_all.py', 'rb') as f:
        code = compile(f.read(), str(root / 'verify_all.py'), 'exec')
        exec(code, {})
