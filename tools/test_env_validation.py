import os
import sys


def main():
    print("=== Environment Validation ===")
    # Python
    print(f"Python: {sys.version.split()[0]}")

    # Torch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
            # Tiny CUDA op
            a = torch.randn(128, 128, device='cuda')
            b = torch.randn(128, 128, device='cuda')
            c = a @ b
            print("CUDA matmul ok:", float(c[0, 0]))
    except Exception as e:
        print("[WARN] Torch check failed:", e)

    # MMCV/MMEngine/MMDet
    try:
        import mmcv
        print("MMCV:", mmcv.__version__)
    except Exception as e:
        print("[WARN] MMCV import failed:", e)

    try:
        import mmengine
        print("MMEngine:", mmengine.__version__)
    except Exception as e:
        print("[WARN] MMEngine import failed:", e)

    try:
        import mmdet
        from mmdet.utils import get_version
        print("MMDetection:", get_version())
    except Exception as e:
        print("[WARN] MMDetection import failed:", e)

    # PyCOCOTools
    try:
        import pycocotools
        print("pycocotools:", getattr(pycocotools, "__version__", "ok"))
    except Exception as e:
        print("[WARN] pycocotools import failed:", e)

    print("=== Done ===")


if __name__ == "__main__":
    main()
