#!/bin/bash
# Harness: evaluate Lean 4 proof attempts against MiniF2F valid set
# Usage: bash run.sh <method_name> "description of what you tried" design_type
#
# Automatically appends score to results.tsv and stages scrubbed copies.

set -e

METHOD=${1:-rrma_lean_v1}
DESCRIPTION=${2:-"no description"}
DESIGN=${3:-tactic}

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
MINIF2F_DIR=${MINIF2F_DIR:-/home/vincent/miniF2F-lean4}
ATTEMPTS_DIR="$DOMAIN_DIR/attempts/$METHOD"
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
LOGS_DIR="$DOMAIN_DIR/logs"

source ~/.elan/env 2>/dev/null || true

mkdir -p "$ATTEMPTS_DIR" "$LOGS_DIR"

PASS=0
FAIL=0
TOTAL=0
AGENT="${CLAUDE_AGENT_ID:-manual}"

echo "[run.sh] Evaluating $METHOD..."

for attempt_file in "$ATTEMPTS_DIR"/*.lean; do
    [ -f "$attempt_file" ] || continue
    problem=$(basename "$attempt_file" .lean)
    TOTAL=$((TOTAL + 1))

    # Use lake env to get Mathlib in scope, timeout 60s per problem
    if timeout 60 bash -c "cd '$MINIF2F_DIR' && lake env lean '$attempt_file'" \
        > /tmp/lean_out_$$.log 2>&1; then
        PASS=$((PASS + 1))
        echo "  PASS: $problem"
    else
        FAIL=$((FAIL + 1))
        grep -m1 "error:" /tmp/lean_out_$$.log 2>/dev/null | sed 's/^/  FAIL: /' || echo "  FAIL: $problem"
    fi
    rm -f /tmp/lean_out_$$.log
done

if [ "$TOTAL" -eq 0 ]; then
    echo "No .lean attempts found in $ATTEMPTS_DIR"
    echo "0.0000"
    exit 0
fi

SCORE=$(python3 -c "print(f'{$PASS/$TOTAL:.4f}')")

echo ""
echo "[run.sh] $METHOD: $PASS / $TOTAL solved"
echo "SCORE=$SCORE"

# ── Auto-append to results.tsv ────────────────────────────────────────────────

# Create header if file doesn't exist
if [ ! -f "$RESULTS_TSV" ]; then
    echo -e "EXP-ID\tscore\tn_attempted\tstatus\tdescription\tagent\tdesign\ttrain_min" > "$RESULTS_TSV"
fi

# Generate next EXP-ID — prefix from instance_name.txt if present
INSTANCE=$(cat "$DOMAIN_DIR/instance_name.txt" 2>/dev/null | tr -d '[:space:]')
if [ -n "$INSTANCE" ] && [ "$INSTANCE" != "nigel" ]; then
    PREFIX=$(echo "$INSTANCE" | sed 's/lean_2nd_generator/lam/; s/lean_//; s/_generator//' | cut -c1-4)
else
    PREFIX="exp"
fi
LAST_N=$(grep -oP "${PREFIX}\K\d+" "$RESULTS_TSV" 2>/dev/null | sort -n | tail -1 || echo 0)
NEXT_N=$(printf "%03d" $((10#$LAST_N + 1)))
EXP_ID="${PREFIX}${NEXT_N}"

# Determine status: keep if score beats current best, else discard
CURRENT_BEST=$(awk -F'\t' 'NR>1 && $4=="keep" {print $2}' "$RESULTS_TSV" 2>/dev/null | sort -rn | head -1)
CURRENT_BEST=${CURRENT_BEST:-0}

STATUS="discard"
if python3 -c "exit(0 if float('$SCORE') > float('$CURRENT_BEST') else 1)" 2>/dev/null; then
    STATUS="keep"
fi

TIMESTAMP=$(date '+%Y-%m-%dT%H:%M:%S')
echo -e "${EXP_ID}\t${SCORE}\t${TOTAL}\t${STATUS}\t${DESCRIPTION}\t${AGENT}\t${DESIGN}\t0" >> "$RESULTS_TSV"

echo ""
echo "[run.sh] Logged: $EXP_ID  score=$SCORE  status=$STATUS  → results.tsv"

# ── Scrub and stage ───────────────────────────────────────────────────────────

TOOLS_DIR="$(cd "$(dirname "$0")" && cd ../.. && pwd)/tools"
SCRUB="$TOOLS_DIR/scrub.py"

if [ -f "$SCRUB" ]; then
    python3 "$SCRUB" "$DOMAIN_DIR" 2>/dev/null && echo "[run.sh] Staged scrubbed copies → staged/"
fi

# Print final score for outer-loop to parse
echo "$SCORE"
