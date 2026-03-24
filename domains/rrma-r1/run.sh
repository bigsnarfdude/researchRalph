#!/bin/bash
# run.sh — training harness. READ ONLY.
# Usage: bash run.sh train.py
# Prints final GSM8K score to stdout.

set -euo pipefail

TRAIN_SCRIPT="${1:-train.py}"
TIMEOUT_MIN=35

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: $TRAIN_SCRIPT not found"
    exit 1
fi

# Run training with timeout
timeout $((TIMEOUT_MIN * 60)) python3 "$TRAIN_SCRIPT" 2>&1 | grep -v "^$" >&2 || {
    code=$?
    if [ $code -eq 124 ]; then
        echo "TIMEOUT after ${TIMEOUT_MIN}m" >&2
    fi
    echo "SCORE: 0.0"
    exit 0
}

# Evaluate checkpoint
if [ -d "./checkpoints/latest" ]; then
    python3 eval.py --model ./checkpoints/latest --samples 200 2>/dev/null | grep "^SCORE:" | awk '{print $2}'
else
    echo "0.0"
fi
