#!/bin/bash
# AuditBench Harness — uses claude -p for investigation and scoring
#
# Usage: bash run.sh [config-file]
# Score: detection rate across target models (0.0 to 1.0, higher is better)
#
# Prerequisites:
#   - claude CLI installed and authenticated
#   - vLLM server running with target model LoRAs loaded
#   - Python 3 with PyYAML

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON="${VENV_PYTHON:-python3}"

# GPU lock for multi-agent sharing (vLLM server is shared)
LOCKFILE="/tmp/auditbench-gpu.lock"
AGENT_ID="${AGENT_ID:-$$}"

gpu_lock() {
    local timeout=1800  # 30 min max wait
    local waited=0
    while ! (set -o noclobber; echo "$AGENT_ID $(date +%s)" > "$LOCKFILE") 2>/dev/null; do
        local started=$(cut -d' ' -f2 "$LOCKFILE" 2>/dev/null)
        local now=$(date +%s)
        local age=$((now - ${started:-0}))
        if [ "$age" -gt 3600 ]; then
            rm -f "$LOCKFILE"
            continue
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo "0.0"
            exit 0
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

gpu_unlock() {
    rm -f "$LOCKFILE"
}

# Check vLLM server is running
check_server() {
    local host=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('host','127.0.0.1'))" 2>/dev/null || echo "127.0.0.1")
    local port=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('port',8192))" 2>/dev/null || echo "8192")
    if ! curl -s "http://${host}:${port}/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server not running at ${host}:${port}" >&2
        echo "Start it with: cd ~/auditing-agents && source .venv/bin/activate && python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --port 8192 --enable-lora" >&2
        echo "0.0"
        exit 0
    fi
}

# Check claude CLI
if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI not found" >&2
    echo "0.0"
    exit 0
fi

trap gpu_unlock EXIT INT TERM

check_server
gpu_lock
$PYTHON "$DIR/engine.py" "$CONFIG" --matches 1 --seed 42
EXIT_CODE=$?
gpu_unlock
trap - EXIT INT TERM
exit $EXIT_CODE
