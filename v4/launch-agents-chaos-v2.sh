#!/bin/bash
# launch-agents-chaos-v2.sh — Fixed Clean Version

DOMAIN_DIR="${1:?Usage: bash launch-agents-chaos-v2.sh <domain> <num-agents> <chaos-ids>}"
NUM_AGENTS="${2:-4}"
CHAOS_IDS="${3:-1,2,3}"
MAX_TURNS="${4:-50}"

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
REPO_ROOT="/Users/vincent/researchRalphLocal"

CHAOS_LOOKUP=",${CHAOS_IDS},"

echo "=== LAUNCHING SWEEP C (3v1) ==="
echo "Domain: $DOMAIN_DIR"

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    SESSION="rrma-worker$i"
    screen -S "$SESSION" -X quit 2>/dev/null

    if [[ "$CHAOS_LOOKUP" == *",$i,"* ]]; then
        # CHAOS AGENT
        echo "  agent$i: CHAOS"
        screen -dmS "$SESSION" bash -c "python3 $REPO_ROOT/rrma_local/tools/chaos_agent_llama.py $DOMAIN_DIR --agent-id $i --turns $MAX_TURNS > $DOMAIN_DIR/logs/agent${i}.log 2>&1"
    else
        # HONEST AGENT
        echo "  agent$i: HONEST (v2.0)"
        screen -dmS "$SESSION" bash -c "python3 $REPO_ROOT/rrma_local/tools/honest_agent_v2_llama.py $DOMAIN_DIR --agent-id $i --turns $MAX_TURNS > $DOMAIN_DIR/logs/agent${i}_v2.log 2>&1"
    fi
    sleep 2
done
echo "=== SWEEP C STARTED ==="
