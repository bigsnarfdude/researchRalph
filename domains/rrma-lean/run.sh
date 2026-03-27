#!/bin/bash
# Harness: evaluate Lean 4 proof attempts against MiniF2F valid set
# Usage: bash run.sh <method_name>
# Score = fraction of attempted problems that compile

set -e
METHOD=${1:-rrma_lean_v1}
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
MINIF2F_DIR=${MINIF2F_DIR:-/home/vincent/miniF2F-lean4}
ATTEMPTS_DIR="$DOMAIN_DIR/attempts/$METHOD"

source ~/.elan/env 2>/dev/null || true

mkdir -p "$ATTEMPTS_DIR"

PASS=0
FAIL=0
TOTAL=0

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
        # Show first error line
        grep -m1 "error:" /tmp/lean_out_$$.log 2>/dev/null | sed 's/^/  FAIL: /' || echo "  FAIL: $problem"
    fi
    rm -f /tmp/lean_out_$$.log
done

if [ "$TOTAL" -eq 0 ]; then
    echo "No .lean attempts found in $ATTEMPTS_DIR"
    echo "SCORE=0.0000"
    exit 0
fi

SCORE=$(python3 -c "print(f'{$PASS/$TOTAL:.4f}')")
echo ""
echo "[run.sh] $METHOD: $PASS / $TOTAL solved"
echo "SCORE=$SCORE"
