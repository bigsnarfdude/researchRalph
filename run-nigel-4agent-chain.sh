#!/bin/bash
set -euo pipefail
cd ~/researchRalph

CHAOS_LAUNCHER="v4/launch-agents-chaos-v2.sh"

wait_done() {
    local domain="$1"
    echo "$(date): Waiting for $domain..."
    while screen -ls 2>/dev/null | grep -q 'rrma-worker'; do
        sleep 60
    done
    echo "$(date): Workers done for $domain"
    bash v4/stop-agents.sh 2>/dev/null || true
    sleep 5
}

# --- 4 agents, 50% chaos (agents 2,3) ---
echo "=== LAUNCHING 4-agent 50% chaos ($(date)) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-nigel-4agent-50 4 "2,3" 200 10
wait_done "4agent-50"
echo "$(date): 4agent-50 results: $(wc -l < domains/nirenberg-1d-chaos-haiku-nigel-4agent-50/results.tsv) lines"

# --- 4 agents, 75% chaos (agents 1,2,3) ---
echo ""
echo "=== LAUNCHING 4-agent 75% chaos ($(date)) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-nigel-4agent-75 4 "1,2,3" 200 10
wait_done "4agent-75"
echo "$(date): 4agent-75 results: $(wc -l < domains/nirenberg-1d-chaos-haiku-nigel-4agent-75/results.tsv) lines"

echo ""
echo "=== ALL DONE $(date) ==="
