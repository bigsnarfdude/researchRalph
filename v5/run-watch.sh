#!/bin/bash
# run-watch.sh — 10-min cadence monitor for the sae-island run (runs on nigel).
# Appends to v5/run_watch.log: per-island rows/best/board-lines, GPU, board-sim.
# Start: nohup bash v5/run-watch.sh > /dev/null 2>&1 &   Stop: pkill -f run-watch.sh

REPO="$HOME/researchRalph"
LOG="$REPO/v5/run_watch.log"

while true; do
    {
        date '+%F %T'
        for i in a b; do
            d="$REPO/domains/sae-island-isl-$i"
            rows=$(awk 'NR>1' "$d/results.tsv" 2>/dev/null | wc -l | tr -d ' ')
            best=$(awk -F'\t' 'NR>1{print $2}' "$d/results.tsv" 2>/dev/null | sort -rn | head -n 1)
            bl=$(wc -l < "$d/blackboard.md" 2>/dev/null | tr -d ' ')
            flag=""
            [ -f "$d/BOARD_OVER_BUDGET" ] && flag=" OVER_BUDGET"
            echo "  isl-$i: exps=${rows:-0} best=${best:-na} board=${bl:-0}L$flag"
        done
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /'
        sim=$(python3 "$REPO/v5/board-sim.py" \
            "$REPO/domains/sae-island-isl-a/blackboard.md" \
            "$REPO/domains/sae-island-isl-b/blackboard.md" 2>/dev/null)
        echo "  board-sim(a,b): ${sim:-na}"
    } >> "$LOG" 2>&1
    sleep 600
done
