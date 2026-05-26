#!/bin/bash
# Harness for imo1993p5 — checks if solution.lean compiles
set -e

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"

# Workspace isolation: prefer agent workspace file
if [ -n "$CLAUDE_AGENT_ID" ] && [ -f "$DOMAIN_DIR/workspace/$CLAUDE_AGENT_ID/solution.lean" ]; then
    SOLUTION="$DOMAIN_DIR/workspace/$CLAUDE_AGENT_ID/solution.lean"
else
    SOLUTION="$DOMAIN_DIR/solution.lean"
fi

export PATH="/home/vincent/.elan/bin:$PATH"

# Check solution exists
if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"
    echo "No solution.lean found"
    exit 0
fi

# Check for sorry
if grep -q "sorry" "$SOLUTION"; then
    echo "SCORE=0.0"
    echo "solution.lean contains sorry"
    exit 0
fi

# Copy to lean project and try to compile
TMP_FILE="$LEAN_PROJECT/Imo1993P5Test.lean"
cp "$SOLUTION" "$TMP_FILE"

cd "$LEAN_PROJECT"
if lake env lean "$TMP_FILE" 2>&1; then
    echo "SCORE=1.0"
    echo "Proof compiles successfully"
    SCORE="1.0"
    STATUS="keep"
else
    echo "SCORE=0.0"
    echo "Proof failed to compile"
    SCORE="0.0"
    STATUS="discard"
fi

# Clean up
rm -f "$TMP_FILE"

# Log to results.tsv
AGENT="${CLAUDE_AGENT_ID:-agent0}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- RH Prevention: flock + chmod ensures only oracle writes results.tsv ---
RESULTS="$DOMAIN_DIR/results.tsv"
LOCK="$DOMAIN_DIR/results.lock"
(
    flock -x -w 30 200 || { echo "[oracle] Could not acquire results.lock — skipping log"; exit 0; }
    chmod 644 "$RESULTS" 2>/dev/null || true
    if [ ! -f "$RESULTS" ] || ! head -1 "$RESULTS" | grep -q "^EXP-ID"; then
        printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "$RESULTS"
    fi
    EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$RESULTS") )))"
    printf "%s\t%s\t%s\t%s\t%s\n" "$EXP_ID" "$SCORE" "$STATUS" "$DESCRIPTION" "$AGENT" >> "$RESULTS"
    chmod 444 "$RESULTS"
) 200>"$LOCK"

# Append to experiments.jsonl
python3 -c "
import json, sys
print(json.dumps({
    'exp_id': '$EXP_ID',
    'score': float('$SCORE'),
    'status': '$STATUS',
    'agent': '$AGENT',
    'timestamp': '$TIMESTAMP',
}))
" >> "$DOMAIN_DIR/experiments.jsonl"
