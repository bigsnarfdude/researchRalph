#!/bin/bash
# run-blind-sweep-cell.sh — Run one cell of the blind sweep experiment
#
# Reads .cell_meta.json for cell parameters, launches agents, monitors
# until target experiments reached, then stops.
#
# Usage: bash v4/run-blind-sweep-cell.sh domains/nirenberg-1d-blind-sweep/B-XX-YY
#
# Archival: after completion, commit and push per protocol.

set -euo pipefail

CELL_DIR="${1:?Usage: bash v4/run-blind-sweep-cell.sh <cell-dir>}"
CELL_DIR="$(cd "$CELL_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

META="$CELL_DIR/.cell_meta.json"
if [ ! -f "$META" ]; then
    echo "Error: no .cell_meta.json in $CELL_DIR"
    echo "Run setup-blind-sweep.sh first."
    exit 1
fi

# Parse metadata
NUM_AGENTS=$(python3 -c "import json; print(json.load(open('$META'))['num_agents'])")
NUM_CHAOS=$(python3 -c "import json; print(json.load(open('$META'))['num_chaos'])")
CHAOS_IDS=$(python3 -c "import json; print(json.load(open('$META'))['chaos_ids'])")
HAS_PRECEDENT=$(python3 -c "import json; print(json.load(open('$META'))['has_precedent'])")
TARGET_EXP=$(python3 -c "import json; print(json.load(open('$META'))['target_experiments'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('$META'))['max_rounds_per_agent'])")
CELL_ID=$(basename "$CELL_DIR")

echo "=== BLIND SWEEP CELL: $CELL_ID ==="
echo "Agents: $NUM_AGENTS | Chaos: $NUM_CHAOS ($CHAOS_IDS) | Precedent: $HAS_PRECEDENT"
echo "Target: $TARGET_EXP experiments | Max rounds/agent: $MAX_ROUNDS"
echo ""

# Refresh context
if [ -f "$REPO_ROOT/tools/refresh_context.py" ]; then
    python3 "$REPO_ROOT/tools/refresh_context.py" "$CELL_DIR" 2>&1 || true
fi

# Launch agents
if [ "$NUM_CHAOS" -gt 0 ]; then
    echo "Launching via chaos v3 (asymmetric briefing)..."
    bash "$SCRIPT_DIR/launch-agents-chaos-v3.sh" "$CELL_DIR" "$NUM_AGENTS" "$CHAOS_IDS" "$MAX_ROUNDS"
else
    echo "Launching via standard launcher (all honest)..."
    bash "$SCRIPT_DIR/launch-agents.sh" "$CELL_DIR" "$NUM_AGENTS" "$MAX_ROUNDS" 2>&1 || \
    bash "$SCRIPT_DIR/launch-agents-chaos-v3.sh" "$CELL_DIR" "$NUM_AGENTS" "" "$MAX_ROUNDS"
fi

echo ""
echo "=== Monitoring until $TARGET_EXP experiments ==="
echo "Check: tail -f $CELL_DIR/results.tsv"
echo ""

# Monitor loop: check experiment count every 30s
START_TIME=$(date +%s)
while true; do
    sleep 30

    # Count experiments (subtract 1 for header)
    EXP_COUNT=$(( $(wc -l < "$CELL_DIR/results.tsv") - 1 ))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))

    # Count branches from secret log
    BRANCHES=$(awk -F'\t' 'NR>1 && $2!="0" {
        if ($2 < -0.5) print "neg";
        else if ($2 > 0.5) print "pos";
        else print "triv"
    }' "$CELL_DIR/.solution_means_secret.tsv" 2>/dev/null | sort -u | wc -l | tr -d ' ')

    # Count crashes
    CRASHES=$(grep -c "crash" "$CELL_DIR/results.tsv" 2>/dev/null || echo 0)

    echo "[${ELAPSED_MIN}m] $CELL_ID: $EXP_COUNT/$TARGET_EXP experiments, $BRANCHES/3 branches, $CRASHES crashes"

    # Check stopping conditions
    if [ "$EXP_COUNT" -ge "$TARGET_EXP" ]; then
        # Check if we need to extend (branch coverage < 3 at 100 exp)
        if [ "$BRANCHES" -lt 3 ] && [ "$EXP_COUNT" -lt 150 ]; then
            echo "  Branch coverage $BRANCHES/3 at $EXP_COUNT exp — extending to 150"
            continue
        fi
        echo ""
        echo "=== TARGET REACHED: $EXP_COUNT experiments ==="
        break
    fi

    # Early stopping: all 3 branches + good residual + >60 experiments
    if [ "$EXP_COUNT" -ge 60 ] && [ "$BRANCHES" -ge 3 ]; then
        BEST=$(awk -F'\t' 'NR>1 && $4=="keep" {print $2}' "$CELL_DIR/results.tsv" | sort -g | head -1)
        if python3 -c "exit(0 if float('$BEST') < 1e-12 else 1)" 2>/dev/null; then
            echo ""
            echo "=== EARLY STOP: 3/3 branches + residual=$BEST < 1e-12 at $EXP_COUNT exp ==="
            break
        fi
    fi

    # Safety: if no new experiments in 10 minutes, something is stuck
    if [ "$ELAPSED" -gt 600 ] && [ "$EXP_COUNT" -lt 5 ]; then
        echo ""
        echo "=== WARNING: only $EXP_COUNT experiments after ${ELAPSED_MIN}m — agents may be stuck ==="
        echo "Check: screen -ls && screen -r rrma-worker0"
        break
    fi

    # Hard timeout: 2 hours per cell
    if [ "$ELAPSED" -gt 7200 ]; then
        echo ""
        echo "=== TIMEOUT: 2 hours, $EXP_COUNT experiments ==="
        break
    fi
done

# Stop agents
echo "Stopping agents..."
bash "$SCRIPT_DIR/stop-agents.sh" 2>/dev/null || true
# Also kill chaos refresh
screen -S rrma-chaos-refresh -X quit 2>/dev/null || true

# Summary
EXP_COUNT=$(( $(wc -l < "$CELL_DIR/results.tsv") - 1 ))
BRANCHES=$(awk -F'\t' 'NR>1 && $2!="0" {
    if ($2 < -0.5) print "neg";
    else if ($2 > 0.5) print "pos";
    else print "triv"
}' "$CELL_DIR/.solution_means_secret.tsv" 2>/dev/null | sort -u | wc -l | tr -d ' ')
CRASHES=$(grep -c "crash" "$CELL_DIR/results.tsv" 2>/dev/null || echo 0)
BEST=$(awk -F'\t' 'NR>1 && $4=="keep" {print $2}' "$CELL_DIR/results.tsv" | sort -g | head -1)

echo ""
echo "=== CELL COMPLETE: $CELL_ID ==="
echo "Experiments: $EXP_COUNT"
echo "Branches found: $BRANCHES/3"
echo "Crashes: $CRASHES"
echo "Best residual: ${BEST:-none}"
echo ""
echo "Archival:"
echo "  cd $REPO_ROOT"
echo "  git add domains/nirenberg-1d-blind-sweep/$CELL_ID/"
echo "  git commit -m 'Blind sweep $CELL_ID: $EXP_COUNT experiments, $BRANCHES branches'"
echo "  git push origin main"
