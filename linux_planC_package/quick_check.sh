#!/bin/bash
# 快速环境检查脚本

echo "=========================================="
echo "Linux服务器环境快速检查"
echo "=========================================="

echo -e "\n[1] 当前工作目录:"
pwd

echo -e "\n[2] Python环境信息:"
which python
python --version
echo "CUDA可用性:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')" 2>/dev/null || echo "✗ 无法检查PyTorch/CUDA"

echo -e "\n[3] MMDetection安装状态:"
python -c "import mmdet; print(f'✓ MMDetection: {mmdet.__version__}')" 2>/dev/null || echo "✗ MMDetection未安装"
python -c "import mmengine; print(f'✓ MMEngine: {mmengine.__version__}')" 2>/dev/null || echo "✗ MMEngine未安装"
python -c "import mmcv; print(f'✓ MMCV: {mmcv.__version__}')" 2>/dev/null || echo "✗ MMCV未安装"

echo -e "\n[4] 常见数据目录:"
for dir in /data /mnt/data /home/msi-kklt/datasets /home/msi-kklt/data; do
    if [ -d "$dir" ]; then
        echo "✓ $dir (存在)"
        ls -lh "$dir" | head -5
    else
        echo "✗ $dir (不存在)"
    fi
done

echo -e "\n[5] MMDetection work_dirs:"
if [ -d ~/mmdetection/work_dirs ]; then
    echo "✓ ~/mmdetection/work_dirs (存在)"
    ls -lh ~/mmdetection/work_dirs/ | head -10
else
    echo "✗ ~/mmdetection/work_dirs (不存在)"
    echo "提示: 可能需要先运行过训练才会创建此目录"
fi

echo -e "\n[6] 当前目录内容:"
ls -lh

echo -e "\n=========================================="
echo "检查完成! 请运行 find_paths.sh 来定位具体文件"
echo "=========================================="
