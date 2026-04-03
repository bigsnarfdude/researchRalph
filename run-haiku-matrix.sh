#!/bin/bash
# run-haiku-matrix.sh — Run remaining Haiku chaos experiments sequentially
# H3 is already running. This script waits for it, then runs H4, H5, H6.
#
# Usage: bash run-haiku-matrix.sh

set -euo pipefail
cd "$(dirname "$0")"

CHAOS_LAUNCHER="v4/launch-agents-chaos-v2.sh"

wait_for_completion() {
    local domain="$1"
    local check_interval=60
    echo "$(date): Waiting for $domain to complete..."
    while true; do
        # Check if any rrma-worker screens are still running
        if ! screen -ls 2>/dev/null | grep -q 'rrma-worker'; then
            echo "$(date): All workers finished for $domain"
            local count=$(wc -l < "$domain/results.tsv")
            echo "$(date): Final experiment count: $count"
            # Kill meta too
            bash v4/stop-agents.sh 2>/dev/null || true
            sleep 5
            return
        fi
        sleep "$check_interval"
    done
}

report() {
    local domain="$1"
    local label="$2"
    echo ""
    echo "=== $label FINAL ==="
    awk -F'\t' '{
        if ($5=="crash" || $4=="crash") c++;
        else if ($4+0 > 0.5) p++;
        else if ($4+0 < -0.5) n++;
        else t++;
        total++
    } END {
        printf "pos=%d (%.1f%%) neg=%d (%.1f%%) triv=%d (%.1f%%) crash=%d total=%d\n",
            p, p/total*100, n, n/total*100, t, t/total*100, c, total
    }' "$domain/results.tsv"
    echo ""
}

# --- H3: already running, just wait ---
echo "=== H3 (4 agents, 25% chaos) — already running ==="
wait_for_completion "domains/nirenberg-1d-chaos-haiku-h3-4agent-25"
report "domains/nirenberg-1d-chaos-haiku-h3-4agent-25" "H3"

# --- H4: 4 agents, 50% chaos ---
echo "=== LAUNCHING H4 (4 agents, 50% chaos) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-h4-4agent-50 4 "2,3" 200 10
wait_for_completion "domains/nirenberg-1d-chaos-haiku-h4-4agent-50"
report "domains/nirenberg-1d-chaos-haiku-h4-4agent-50" "H4"

# --- H5: 8 agents, 12.5% chaos ---
echo "=== LAUNCHING H5 (8 agents, 12.5% chaos) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-h5-8agent-12 8 "7" 200 10
wait_for_completion "domains/nirenberg-1d-chaos-haiku-h5-8agent-12"
report "domains/nirenberg-1d-chaos-haiku-h5-8agent-12" "H5"

# --- H6: 8 agents, 37.5% chaos ---
echo "=== LAUNCHING H6 (8 agents, 37.5% chaos) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-h6-8agent-37 8 "5,6,7" 200 10
wait_for_completion "domains/nirenberg-1d-chaos-haiku-h6-8agent-37"
report "domains/nirenberg-1d-chaos-haiku-h6-8agent-37" "H6"

echo ""
echo "=== ALL HAIKU EXPERIMENTS COMPLETE ==="
echo ""
echo "Summary:"
report "domains/nirenberg-1d-chaos-haiku-h1-control" "H1 (control, 0%)"
report "domains/nirenberg-1d-chaos-haiku" "H2 (2-agent, 50%)"
report "domains/nirenberg-1d-chaos-haiku-h3-4agent-25" "H3 (4-agent, 25%)"
report "domains/nirenberg-1d-chaos-haiku-h4-4agent-50" "H4 (4-agent, 50%)"
report "domains/nirenberg-1d-chaos-haiku-h5-8agent-12" "H5 (8-agent, 12.5%)"
report "domains/nirenberg-1d-chaos-haiku-h6-8agent-37" "H6 (8-agent, 37.5%)"
