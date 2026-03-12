#!/bin/bash
# SAE-Bench Harness — trains SAE and outputs F1 score
#
# Usage: bash run.sh [config-file]
# Score: F1 on SynthSAEBench-16k (0.0 to 1.0, higher is better)

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Use venv python if available (for GPU dependencies)
PYTHON="${VENV_PYTHON:-python3}"
if [ -f "$HOME/venv/bin/python3" ]; then
    PYTHON="$HOME/venv/bin/python3"
fi

# GPU lock for multi-agent single-GPU sharing
LOCKFILE="/tmp/saebench-gpu.lock"
AGENT_ID="${AGENT_ID:-$$}"

gpu_lock() {
    local timeout=600  # 10 min max wait
    local waited=0
    while ! (set -o noclobber; echo "$AGENT_ID $(date +%s)" > "$LOCKFILE") 2>/dev/null; do
        local started=$(cut -d' ' -f2 "$LOCKFILE" 2>/dev/null)
        local now=$(date +%s)
        local age=$((now - ${started:-0}))
        # Break stale locks (>20 min — training can take that long)
        if [ "$age" -gt 1200 ]; then
            rm -f "$LOCKFILE"
            continue
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo "0.0"  # timeout = score 0
            exit 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

gpu_unlock() {
    rm -f "$LOCKFILE"
}

gpu_lock
$PYTHON "$DIR/engine.py" "$CONFIG" --matches 1 --seed 42
EXIT_CODE=$?
gpu_unlock
exit $EXIT_CODE
