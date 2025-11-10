#!/bin/bash
# Training Monitor Script for MMDetection Stage2
# Usage: ./monitor_training.sh [work_dir] [refresh_seconds]
# Example: ./monitor_training.sh work_dirs/stage2_kaist_full_conservative_remote 30

WORK_DIR="${1:-work_dirs/stage2_kaist_full_conservative_remote}"
REFRESH_SEC="${2:-30}"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print header
print_header() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   MMDetection Stage2 Training Monitor${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Work Dir: ${GREEN}$WORK_DIR${NC}"
    echo -e "Refresh: ${YELLOW}${REFRESH_SEC}s${NC}  |  Press Ctrl+C to exit"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Function to extract latest metrics
extract_metrics() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo -e "${RED}ERROR: Log file not found: $log_file${NC}"
        return 1
    fi
    
    # Get latest epoch number
    local epoch=$(grep -oP 'Epoch\(val\)\s+\[\d+\]\[\d+/\d+\]' "$log_file" | tail -n1 | grep -oP '\[\K\d+(?=\])' | head -n1)
    
    # Get latest mAP
    local map=$(grep -oP 'pascal_voc/mAP:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest recall
    local recall=$(grep -oP 'pascal_voc/AP50:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest domain_weight
    local domain_weight=$(grep -oP 'domain_weight.*?:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest loss
    local loss=$(grep -oP 'loss:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest loss_cls
    local loss_cls=$(grep -oP 'loss_cls:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest loss_bbox
    local loss_bbox=$(grep -oP 'loss_bbox:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    # Get latest learning rate
    local lr=$(grep -oP 'lr:\s+\K[\d.e-]+' "$log_file" | tail -n1)
    
    # Print metrics
    echo -e "${GREEN}Current Epoch:${NC} ${epoch:-N/A}"
    echo ""
    echo -e "${YELLOW}=== Validation Metrics ===${NC}"
    echo -e "  mAP:         ${GREEN}${map:-N/A}${NC}"
    echo -e "  AP50:        ${map:+${GREEN}}${recall:-N/A}${map:+${NC}}"
    echo ""
    echo -e "${YELLOW}=== Training Losses ===${NC}"
    echo -e "  Total Loss:  ${loss:-N/A}"
    echo -e "  Loss Cls:    ${loss_cls:-N/A}"
    echo -e "  Loss BBox:   ${loss_bbox:-N/A}"
    echo ""
    echo -e "${YELLOW}=== Hyperparameters ===${NC}"
    echo -e "  Domain Weight: ${domain_weight:-N/A}"
    echo -e "  Learning Rate: ${lr:-N/A}"
    echo ""
}

# Function to show recent errors
show_errors() {
    local log_file="$1"
    echo -e "${RED}=== Recent Errors/Warnings ===${NC}"
    grep -iE 'error|warning|exception' "$log_file" | tail -n 5 || echo "  No recent errors"
    echo ""
}

# Function to show training progress bar
show_progress() {
    local log_file="$1"
    local max_epochs=12
    
    # Extract current epoch
    local current_epoch=$(grep -oP 'Epoch\(train\)\s+\[\K\d+' "$log_file" | tail -n1)
    
    if [ -z "$current_epoch" ]; then
        echo -e "${YELLOW}Training not started yet${NC}"
        return
    fi
    
    # Calculate progress
    local progress=$((current_epoch * 100 / max_epochs))
    local filled=$((progress / 5))
    local empty=$((20 - filled))
    
    # Build progress bar
    local bar=$(printf "%${filled}s" | tr ' ' '█')
    local space=$(printf "%${empty}s" | tr ' ' '░')
    
    echo -e "${YELLOW}=== Training Progress ===${NC}"
    echo -e "  Epoch: ${current_epoch}/${max_epochs}  [${bar}${space}] ${progress}%"
    echo ""
}

# Function to check decision points
check_decision_points() {
    local log_file="$1"
    local epoch=$(grep -oP 'Epoch\(val\)\s+\[\K\d+' "$log_file" | tail -n1)
    local map=$(grep -oP 'pascal_voc/mAP:\s+\K[\d.]+' "$log_file" | tail -n1)
    
    if [ -z "$epoch" ] || [ -z "$map" ]; then
        return
    fi
    
    echo -e "${BLUE}=== Decision Points ===${NC}"
    
    # Epoch 4 check
    if [ "$epoch" -eq 4 ]; then
        if (( $(echo "$map < 0.58" | bc -l) )); then
            echo -e "${YELLOW}⚠ Epoch 4: mAP < 0.58 - Monitor domain_weight${NC}"
        fi
    fi
    
    # Epoch 6 check
    if [ "$epoch" -eq 6 ]; then
        if (( $(echo "$map < 0.60" | bc -l) )); then
            echo -e "${RED}⚠ Epoch 6: mAP < 0.60 - Consider switching to Plan C${NC}"
            echo -e "  Command: python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
            echo -e "           --work-dir work_dirs/stage2_kaist_full_C \\"
            echo -e "           --cfg-options custom_hooks.0.target_domain_weight=0.06 \\"
            echo -e "                         custom_hooks.0.warmup_epochs=6"
        fi
    fi
    
    # Epoch 12 check
    if [ "$epoch" -eq 12 ]; then
        if (( $(echo "$map >= 0.70" | bc -l) )); then
            echo -e "${GREEN}✓ Epoch 12: mAP >= 0.70 - Can try Plan B for further improvement${NC}"
            echo -e "  Command: python tools/train.py configs/llvip/stage2_kaist_full_conservative.py \\"
            echo -e "           --work-dir work_dirs/stage2_kaist_full_B \\"
            echo -e "           --cfg-options optim_wrapper.optimizer.lr=0.00035 \\"
            echo -e "                         custom_hooks.0.target_domain_weight=0.12"
        fi
    fi
    echo ""
}

# Function to show GPU utilization
show_gpu() {
    echo -e "${YELLOW}=== GPU Status ===${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s (%s): %s%% GPU, %s/%s MB\n", $1, $2, $3, $4, $5}'
    echo ""
}

# Main monitoring loop
main() {
    # Find latest log file
    local log_file=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    
    if [ -z "$log_file" ]; then
        echo -e "${RED}ERROR: No log files found in $WORK_DIR${NC}"
        echo "Waiting for training to start..."
        sleep 5
        return
    fi
    
    print_header
    echo -e "Log File: ${GREEN}$(basename $log_file)${NC}"
    echo ""
    
    show_gpu
    show_progress "$log_file"
    extract_metrics "$log_file"
    check_decision_points "$log_file"
    show_errors "$log_file"
    
    echo -e "${BLUE}Last updated: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
}

# Trap Ctrl+C
trap 'echo -e "\n${GREEN}Monitor stopped.${NC}"; exit 0' INT

# Check dependencies
command -v bc >/dev/null 2>&1 || { echo "Installing bc..."; sudo apt-get install -y bc; }

# Run loop
while true; do
    main
    sleep "$REFRESH_SEC"
done
