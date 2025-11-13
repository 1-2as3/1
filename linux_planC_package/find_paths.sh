#!/bin/bash
# 路径发现脚本 - 在Linux服务器上定位KAIST数据集和checkpoint

echo "=========================================="
echo "正在查找KAIST数据集..."
echo "=========================================="

# 查找KAIST数据集目录
echo -e "\n[1] 搜索常见数据集目录..."
for dir in /data /mnt/data /home/msi-kklt/datasets /home/msi-kklt/data; do
    if [ -d "$dir" ]; then
        echo "✓ 检查: $dir"
        find "$dir" -maxdepth 3 -type d \( -iname "*kaist*" -o -iname "*llvip*" \) 2>/dev/null
    fi
done

echo -e "\n[2] 搜索包含'visible'和'infrared'子目录的目录..."
for dir in /data /mnt/data /home/msi-kklt; do
    if [ -d "$dir" ]; then
        find "$dir" -maxdepth 4 -type d -name "visible" 2>/dev/null | while read vdir; do
            parent=$(dirname "$vdir")
            if [ -d "$parent/infrared" ]; then
                echo "✓ 找到可能的数据集: $parent"
            fi
        done
    fi
done

echo -e "\n=========================================="
echo "正在查找Stage 1 checkpoint文件..."
echo "=========================================="

# 查找checkpoint文件
echo -e "\n[3] 搜索.pth checkpoint文件..."
find ~/mmdetection/work_dirs -name "*.pth" 2>/dev/null | head -20

echo -e "\n[4] 搜索包含'stage1'或'epoch'的checkpoint..."
find ~/ -name "*stage1*.pth" -o -name "*epoch_48*.pth" -o -name "*backup*.pth" 2>/dev/null | head -20

echo -e "\n[5] 检查常见work_dirs结构..."
if [ -d ~/mmdetection/work_dirs ]; then
    echo "✓ work_dirs目录存在,内容如下:"
    ls -lh ~/mmdetection/work_dirs/ 2>/dev/null
else
    echo "✗ ~/mmdetection/work_dirs 不存在"
fi

echo -e "\n[6] 搜索最近修改的.pth文件(可能是训练checkpoint)..."
find ~/ -name "*.pth" -type f -mtime -30 2>/dev/null | head -10

echo -e "\n=========================================="
echo "路径发现完成!"
echo "=========================================="
echo -e "\n请根据上述输出,找到正确的路径后,运行:"
echo "bash setup_planC.sh <实际的KAIST数据集路径> <实际的checkpoint路径>"
echo -e "\n示例:"
echo "bash setup_planC.sh /mnt/data/KAIST ~/mmdetection/work_dirs/stage2_1_pure_detection/stage2_1_backup_ep2.pth"
