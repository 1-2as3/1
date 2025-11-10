#!/bin/bash
# Quick Fix Script for mmcv._ext Import Error
# Run this on remote server if mmcv installation failed
# Usage: bash fix_mmcv.sh

set -e

echo "=========================================="
echo "  mmcv CUDA Extension Fix Utility"
echo "=========================================="
echo ""

# Activate environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Current conda env: $CONDA_DEFAULT_ENV"
else
    echo "Activating mmdet_py311..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mmdet_py311 || { echo "ERROR: Cannot activate env"; exit 1; }
fi

echo ""
echo "Step 1: Diagnose current state"
echo "================================"

echo "PyTorch version:"
python -c "import torch; print('  Torch:', torch.__version__, 'CUDA:', torch.version.cuda)" || echo "  ERROR: PyTorch not installed"

echo "mmcv installation:"
python -c "import mmcv; print('  mmcv:', mmcv.__version__, '\n  Path:', mmcv.__file__)" 2>&1 || echo "  ERROR: mmcv not found"

echo "Checking for multiple mmcv versions:"
pip list | grep mmcv

echo ""
echo "Attempting to import mmcv._ext..."
python -c "import mmcv._ext; print('  ✓ _ext module OK')" 2>&1 || {
    echo "  ✗ _ext module FAILED (expected)"
    echo ""
    echo "Step 2: Apply Fix"
    echo "=================="
    
    # Method 1: mim install (highest success rate)
    echo ""
    echo "Method 1: Using mim (OpenMMLab package manager)..."
    pip install openmim -q || { echo "WARNING: openmim install failed"; }
    
    if command -v mim &> /dev/null; then
        echo "  Uninstalling old mmcv..."
        pip uninstall mmcv mmcv-full -y 2>/dev/null || true
        
        echo "  Installing mmcv 2.0.1 via mim..."
        mim install mmcv==2.0.1 || {
            echo "  WARNING: mim install 2.0.1 failed, trying 2.0.0..."
            mim install mmcv==2.0.0 || {
                echo "  ERROR: mim install failed"
                METHOD1_SUCCESS=false
            }
        }
        
        echo "  Verifying..."
        python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4).cuda(), torch.rand(5).cuda(), 0.5); print('  ✓ Method 1 SUCCESS')" 2>&1 && {
            echo ""
            echo "=========================================="
            echo "  ✓ mmcv FIXED via mim!"
            echo "=========================================="
            exit 0
        } || {
            echo "  ✗ Method 1 verification failed"
            METHOD1_SUCCESS=false
        }
    else
        echo "  WARNING: mim not available"
        METHOD1_SUCCESS=false
    fi
    
    # Method 2: Direct install mmcv 2.0.0
    if [ "$METHOD1_SUCCESS" = false ]; then
        echo ""
        echo "Method 2: Direct pip install mmcv 2.0.0..."
        pip uninstall mmcv mmcv-full -y 2>/dev/null || true
        pip install mmcv==2.0.0 --no-cache-dir || {
            echo "  ERROR: mmcv 2.0.0 install failed"
            METHOD2_SUCCESS=false
        }
        
        echo "  Verifying..."
        python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4).cuda(), torch.rand(5).cuda(), 0.5); print('  ✓ Method 2 SUCCESS')" 2>&1 && {
            echo ""
            echo "=========================================="
            echo "  ✓ mmcv FIXED via direct install!"
            echo "=========================================="
            exit 0
        } || {
            echo "  ✗ Method 2 verification failed"
            METHOD2_SUCCESS=false
        }
    fi
    
    # Method 3: Downgrade PyTorch to 2.0.1
    if [ "$METHOD2_SUCCESS" = false ]; then
        echo ""
        echo "Method 3: Downgrade PyTorch to 2.0.1 + mmcv 2.0.1..."
        read -p "This will reinstall PyTorch. Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip uninstall torch torchvision torchaudio mmcv mmcv-full -y
            pip install --index-url https://download.pytorch.org/whl/cu118 \
                torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
            pip install mmcv==2.0.1 --no-cache-dir
            
            echo "  Verifying..."
            python -c "import torch; print('  PyTorch:', torch.__version__)"
            python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4).cuda(), torch.rand(5).cuda(), 0.5); print('  ✓ Method 3 SUCCESS')" 2>&1 && {
                echo ""
                echo "=========================================="
                echo "  ✓ mmcv FIXED via PyTorch downgrade!"
                echo "=========================================="
                exit 0
            } || {
                echo "  ✗ Method 3 verification failed"
            }
        else
            echo "  Skipped Method 3"
        fi
    fi
    
    # Method 4: Compile from source (last resort)
    echo ""
    echo "Method 4: Compile mmcv from source (takes 10-20 min)..."
    read -p "Attempt source compilation? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Installing build dependencies..."
        sudo apt-get update -qq
        sudo apt-get install -y build-essential ninja-build -qq
        
        echo "  Installing CUDA toolkit..."
        conda install -c conda-forge cudatoolkit-dev=11.8 -y -q || echo "  WARNING: cudatoolkit-dev install failed"
        
        echo "  Compiling mmcv (this will take a while)..."
        pip uninstall mmcv mmcv-full -y 2>/dev/null || true
        export MMCV_WITH_OPS=1
        export FORCE_CUDA=1
        pip install mmcv==2.0.1 --no-binary mmcv -v
        
        echo "  Verifying..."
        python -c "from mmcv.ops import nms; import torch; nms(torch.randn(5,4).cuda(), torch.rand(5).cuda(), 0.5); print('  ✓ Method 4 SUCCESS')" 2>&1 && {
            echo ""
            echo "=========================================="
            echo "  ✓ mmcv FIXED via source compilation!"
            echo "=========================================="
            exit 0
        } || {
            echo "  ✗ Method 4 verification failed"
        }
    else
        echo "  Skipped Method 4"
    fi
    
    # All methods failed
    echo ""
    echo "=========================================="
    echo "  ✗ ALL METHODS FAILED"
    echo "=========================================="
    echo ""
    echo "Please check REMOTE_DEPLOY_TROUBLESHOOTING.md Issue 3 for manual steps."
    echo "Diagnostic info saved to mmcv_debug.txt"
    
    {
        echo "=== System Info ==="
        uname -a
        echo ""
        echo "=== CUDA Driver ==="
        nvidia-smi
        echo ""
        echo "=== Python Packages ==="
        pip list | grep -E 'torch|mmcv|mmdet|mmengine'
        echo ""
        echo "=== PyTorch Info ==="
        python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDNN:', torch.backends.cudnn.version())"
        echo ""
        echo "=== mmcv Import Error ==="
        python -c "from mmcv.ops import nms" 2>&1
    } > mmcv_debug.txt
    
    exit 1
}

echo ""
echo "=========================================="
echo "  ✓ mmcv already working correctly!"
echo "=========================================="
