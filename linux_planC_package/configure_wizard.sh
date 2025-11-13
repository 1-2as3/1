#!/bin/bash
# 交互式路径配置向导 - 自动发现并配置Plan C

set -e

echo "=========================================="
echo "Plan C 路径配置向导"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 第1步: 发现KAIST数据集
echo -e "${YELLOW}[1/4] 正在搜索KAIST数据集...${NC}"
KAIST_CANDIDATES=()

# 搜索策略1: 查找包含visible和infrared的目录
while IFS= read -r line; do
    KAIST_CANDIDATES+=("$line")
done < <(find /data /mnt /home/msi-kklt -maxdepth 4 -type d -name "visible" 2>/dev/null | while read vdir; do
    parent=$(dirname "$vdir")
    if [ -d "$parent/infrared" ] || [ -d "$parent/lwir" ]; then
        echo "$parent"
    fi
done | sort -u)

# 搜索策略2: 查找名称包含kaist的目录
while IFS= read -r line; do
    if [[ ! " ${KAIST_CANDIDATES[@]} " =~ " ${line} " ]]; then
        KAIST_CANDIDATES+=("$line")
    fi
done < <(find /data /mnt /home/msi-kklt -maxdepth 4 -type d \( -iname "*kaist*" -o -iname "*llvip*" \) 2>/dev/null)

if [ ${#KAIST_CANDIDATES[@]} -eq 0 ]; then
    echo -e "${RED}✗ 未找到KAIST数据集!${NC}"
    echo "请手动输入数据集路径:"
    read -p "KAIST数据集路径: " KAIST_PATH
else
    echo -e "${GREEN}✓ 找到 ${#KAIST_CANDIDATES[@]} 个候选位置:${NC}"
    for i in "${!KAIST_CANDIDATES[@]}"; do
        echo "  [$i] ${KAIST_CANDIDATES[$i]}"
        # 显示目录内容预览
        if [ -d "${KAIST_CANDIDATES[$i]}" ]; then
            ls -1 "${KAIST_CANDIDATES[$i]}" | head -3 | sed 's/^/      /'
        fi
    done
    
    # 自动选择最可能的路径
    BEST_CANDIDATE=""
    for candidate in "${KAIST_CANDIDATES[@]}"; do
        if [ -d "$candidate/visible" ] && [ -d "$candidate/infrared" ] && [ -d "$candidate/annotations" ]; then
            BEST_CANDIDATE="$candidate"
            break
        elif [ -d "$candidate/visible" ] && [ -d "$candidate/lwir" ]; then
            BEST_CANDIDATE="$candidate"
            break
        fi
    done
    
    if [ -n "$BEST_CANDIDATE" ]; then
        echo -e "${GREEN}✓ 推荐使用: $BEST_CANDIDATE${NC}"
        read -p "使用推荐路径? (y/n, 默认y): " use_best
        if [ -z "$use_best" ] || [ "$use_best" = "y" ]; then
            KAIST_PATH="$BEST_CANDIDATE"
        else
            read -p "请选择编号或输入自定义路径: " choice
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -lt ${#KAIST_CANDIDATES[@]} ]; then
                KAIST_PATH="${KAIST_CANDIDATES[$choice]}"
            else
                KAIST_PATH="$choice"
            fi
        fi
    else
        read -p "请选择编号或输入自定义路径: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -lt ${#KAIST_CANDIDATES[@]} ]; then
            KAIST_PATH="${KAIST_CANDIDATES[$choice]}"
        else
            KAIST_PATH="$choice"
        fi
    fi
fi

# 验证数据集路径
if [ ! -d "$KAIST_PATH" ]; then
    echo -e "${RED}✗ 错误: 路径不存在: $KAIST_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ KAIST数据集路径: $KAIST_PATH${NC}"

# 第2步: 发现checkpoint文件
echo -e "\n${YELLOW}[2/4] 正在搜索checkpoint文件...${NC}"
CHECKPOINT_CANDIDATES=()

# 搜索策略1: 查找work_dirs中的.pth文件
if [ -d ~/mmdetection/work_dirs ]; then
    while IFS= read -r line; do
        CHECKPOINT_CANDIDATES+=("$line")
    done < <(find ~/mmdetection/work_dirs -name "*.pth" -type f 2>/dev/null)
fi

# 搜索策略2: 查找包含stage1或epoch_48的文件
while IFS= read -r line; do
    if [[ ! " ${CHECKPOINT_CANDIDATES[@]} " =~ " ${line} " ]]; then
        CHECKPOINT_CANDIDATES+=("$line")
    fi
done < <(find ~/ -name "*stage1*.pth" -o -name "*epoch_48*.pth" -o -name "*backup*.pth" 2>/dev/null | head -20)

if [ ${#CHECKPOINT_CANDIDATES[@]} -eq 0 ]; then
    echo -e "${YELLOW}! 未找到checkpoint文件${NC}"
    echo "选项:"
    echo "  1. 输入checkpoint路径"
    echo "  2. 使用MMDetection预训练模型"
    echo "  3. 从头训练 (不推荐)"
    read -p "请选择 (1-3): " ckpt_choice
    
    case $ckpt_choice in
        1)
            read -p "Checkpoint路径: " CHECKPOINT_PATH
            ;;
        2)
            CHECKPOINT_PATH="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
            echo -e "${GREEN}✓ 将使用COCO预训练模型${NC}"
            ;;
        3)
            CHECKPOINT_PATH=""
            echo -e "${YELLOW}! 将从头训练${NC}"
            ;;
        *)
            echo -e "${RED}✗ 无效选择${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}✓ 找到 ${#CHECKPOINT_CANDIDATES[@]} 个checkpoint:${NC}"
    for i in "${!CHECKPOINT_CANDIDATES[@]}"; do
        ckpt="${CHECKPOINT_CANDIDATES[$i]}"
        size=$(ls -lh "$ckpt" | awk '{print $5}')
        echo "  [$i] $ckpt ($size)"
    done
    
    # 自动选择最近的checkpoint
    LATEST_CHECKPOINT=$(ls -t "${CHECKPOINT_CANDIDATES[@]}" 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo -e "${GREEN}✓ 推荐使用最新: $LATEST_CHECKPOINT${NC}"
        read -p "使用推荐checkpoint? (y/n, 默认y): " use_latest
        if [ -z "$use_latest" ] || [ "$use_latest" = "y" ]; then
            CHECKPOINT_PATH="$LATEST_CHECKPOINT"
        else
            read -p "请选择编号或输入自定义路径 (留空=从头训练): " choice
            if [ -z "$choice" ]; then
                CHECKPOINT_PATH=""
            elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -lt ${#CHECKPOINT_CANDIDATES[@]} ]; then
                CHECKPOINT_PATH="${CHECKPOINT_CANDIDATES[$choice]}"
            else
                CHECKPOINT_PATH="$choice"
            fi
        fi
    fi
fi

# 验证checkpoint路径
if [ -n "$CHECKPOINT_PATH" ] && [[ ! "$CHECKPOINT_PATH" =~ ^https?:// ]] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}✗ 错误: Checkpoint文件不存在: $CHECKPOINT_PATH${NC}"
    exit 1
fi

if [ -n "$CHECKPOINT_PATH" ]; then
    echo -e "${GREEN}✓ Checkpoint路径: $CHECKPOINT_PATH${NC}"
else
    echo -e "${YELLOW}! 将从头训练 (无预训练checkpoint)${NC}"
fi

# 第3步: 更新配置文件
echo -e "\n${YELLOW}[3/4] 正在更新配置文件...${NC}"

if [ ! -f "config_planC_linux.py" ]; then
    echo -e "${RED}✗ 错误: config_planC_linux.py 不存在${NC}"
    exit 1
fi

# 备份原配置
cp config_planC_linux.py config_planC_linux.py.bak

# 更新data_root
sed -i "s|data_root = .*|data_root = '$KAIST_PATH/'|g" config_planC_linux.py

# 更新load_from
if [ -n "$CHECKPOINT_PATH" ]; then
    sed -i "s|load_from = .*|load_from = '$CHECKPOINT_PATH'|g" config_planC_linux.py
else
    sed -i "s|load_from = .*|# load_from = None  # Train from scratch|g" config_planC_linux.py
fi

echo -e "${GREEN}✓ 配置文件已更新${NC}"
echo "  - data_root = '$KAIST_PATH/'"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  - load_from = '$CHECKPOINT_PATH'"
else
    echo "  - load_from = None (从头训练)"
fi

# 第4步: 运行烟雾测试
echo -e "\n${YELLOW}[4/4] 正在运行双模态数据加载测试...${NC}"

if python test_dual_modality.py config_planC_linux.py; then
    echo -e "${GREEN}✓ 烟雾测试通过!${NC}"
    echo ""
    echo "=========================================="
    echo -e "${GREEN}配置完成! 可以开始训练${NC}"
    echo "=========================================="
    echo ""
    echo "启动训练:"
    echo "  bash train_planC.sh           # 单GPU"
    echo "  bash train_planC.sh 0,1       # 双GPU"
    echo ""
    echo "监控训练:"
    echo "  tail -f work_dirs/planC_*/train_*.log | grep -E 'loss_macl|mAP'"
    echo ""
else
    echo -e "${RED}✗ 烟雾测试失败!${NC}"
    echo "请检查:"
    echo "  1. 数据集结构是否正确"
    echo "  2. visible/ 和 infrared/ 目录是否存在"
    echo "  3. annotations/ 目录是否包含train.txt和test.txt"
    echo ""
    echo "回滚配置:"
    echo "  mv config_planC_linux.py.bak config_planC_linux.py"
    exit 1
fi
