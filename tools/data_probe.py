"""Quick dataset probe: read first N lines of ann_file(s) and try opening images.

Supports simple line formats:
- Single path per line: <rel_path>
- Two paths per line: <rel_path_visible> <rel_path_infrared>

Usage examples (Windows paths need escaping or quotes):
  python tools/data_probe.py --ann C:\\KAIST_PROCESSED\\ImageSets\\train.txt --root C:\\KAIST_PROCESSED --limit 3
  python tools/data_probe.py --ann C:\\LLVIP\\LLVIP\\ImageSets\\train.txt --root C:\\LLVIP\\LLVIP --limit 3
  python tools/data_probe.py --ann C:\\M3FD\\cleaned_dataset\\train.txt --root C:\\M3FD\\cleaned_dataset --limit 3
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_line(line: str) -> List[str]:
    line = line.strip()
    if not line or line.startswith('#'):
        return []
    # try whitespace split first
    parts = line.split()
    # If looks like CSV with commas and no spaces
    if len(parts) == 1 and ',' in parts[0]:
        parts = [p.strip() for p in parts[0].split(',') if p.strip()]
    # If looks like JSON-ish, last resort: strip quotes/brackets
    if len(parts) == 1 and (parts[0].startswith('[') or parts[0].startswith('{')):
        try:
            import json
            obj = json.loads(parts[0])
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
    return parts


def try_open_image(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, 'File not found'
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True, 'PIL ok'
    except Exception as e1:
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                return False, 'cv2 failed to read (None)'
            return True, 'cv2 ok'
        except Exception as e2:
            return False, f'PIL:{type(e1).__name__} {e1}; cv2:{type(e2).__name__} {e2}'


def make_kaist_paths(root: Path, token: str) -> List[Path]:
    # token like set00_V000_lwir_I01216 (no extension)
    paths = []
    stem = token
    if stem.lower().endswith('.jpg'):
        stem = stem[:-4]
    lwir_name = stem + '.jpg'
    vis_name = stem.replace('lwir', 'visible') + '.jpg'
    paths.append(root / 'infrared' / lwir_name)
    paths.append(root / 'visible' / vis_name)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='Annotation list file (one or two relative paths per line)')
    ap.add_argument('--root', required=True, help='Dataset root directory to join with relative paths')
    ap.add_argument('--limit', type=int, default=3)
    args = ap.parse_args()

    ann = Path(args.ann)
    root = Path(args.root)
    if not ann.exists():
        print('[FAIL] ann file not found:', ann)
        return
    if not root.exists():
        print('[FAIL] root not found:', root)
        return

    lines = ann.read_text(encoding='utf-8', errors='ignore').splitlines()
    print(f'[INFO] ann: {ann} total lines: {len(lines)}')
    checked = 0
    for i, line in enumerate(lines):
        parts = parse_line(line)
        if not parts:
            continue
        paths = []
        if len(parts) >= 2:
            paths = [root / parts[0], root / parts[1]]
        else:
            # Single token: try direct join, else KAIST-style fallback
            direct = root / parts[0]
            if direct.exists():
                paths = [direct]
            else:
                # Try KAIST convention
                paths = make_kaist_paths(root, parts[0])
        print(f'Line {i+1}: raw={parts}')
        for p in paths:
            ok, msg = try_open_image(p)
            print('  ->', p, '|', 'OK' if ok else 'FAIL', '|', msg)
        checked += 1
        if checked >= args.limit:
            break
    if checked == 0:
        print('[WARN] no valid lines parsed in ann file')


if __name__ == '__main__':
    main()
