#!/bin/bash
# Quick Log Analyzer for MMDetection Training
# Extract key metrics and plot trends (requires gnuplot)
# Usage: ./analyze_logs.sh <work_dir>

WORK_DIR="${1:-work_dirs/stage2_kaist_full_conservative_remote}"
LOG_FILE=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")

if [ -z "$LOG_FILE" ]; then
    echo "ERROR: No log file found in $WORK_DIR"
    exit 1
fi

echo "Analyzing: $LOG_FILE"
echo ""

# Extract epoch-wise metrics
echo "=== Epoch Summary ==="
grep -E 'Epoch\(val\).*pascal_voc/mAP' "$LOG_FILE" | \
    grep -oP 'Epoch\(val\)\s+\[\K\d+|\bpascal_voc/mAP:\s+\K[\d.]+|\bdomain_weight.*?:\s+\K[\d.]+' | \
    paste - - - | \
    awk '{printf "Epoch %2d: mAP=%.4f  domain_weight=%.4f\n", $1, $2, $3}'

echo ""
echo "=== Training Loss Trend (Last 10 epochs) ==="
grep -oP 'Epoch\(train\)\s+\[\K\d+.*?loss:\s+[\d.]+' "$LOG_FILE" | tail -n 10 | \
    awk '{match($0, /([0-9]+).*loss:\s+([0-9.]+)/, arr); printf "Epoch %2d: loss=%.4f\n", arr[1], arr[2]}'

echo ""
echo "=== Best Checkpoint ==="
if [ -f "$WORK_DIR/last_checkpoint" ]; then
    echo "Latest checkpoint: $(cat $WORK_DIR/last_checkpoint)"
fi
ls -lh "$WORK_DIR"/best_*.pth 2>/dev/null | awk '{print "Best model:", $9, "(" $5 ")"}'

echo ""
echo "=== Error Summary ==="
grep -iE 'error|exception' "$LOG_FILE" | wc -l | xargs echo "Total errors:"
grep -iE 'warning' "$LOG_FILE" | wc -l | xargs echo "Total warnings:"

echo ""
echo "=== Recent Issues (Last 5) ==="
grep -iE 'error|warning|exception' "$LOG_FILE" | tail -n 5

echo ""
echo "=== Generate CSV for external plotting ==="
CSV_FILE="${WORK_DIR}/metrics_$(date +%Y%m%d_%H%M%S).csv"
echo "epoch,mAP,domain_weight,loss" > "$CSV_FILE"
grep -E 'Epoch\(val\)' "$LOG_FILE" | \
    grep -oP 'Epoch\(val\)\s+\[\K\d+|\bpascal_voc/mAP:\s+\K[\d.]+|\bdomain_weight.*?:\s+\K[\d.]+' | \
    paste - - - | \
    awk '{printf "%d,%.6f,%.6f,0\n", $1, $2, $3}' >> "$CSV_FILE"

echo "CSV saved to: $CSV_FILE"
echo "You can plot this with: gnuplot -e \"set datafile separator ','; plot '$CSV_FILE' using 1:2 with lines title 'mAP'\""
