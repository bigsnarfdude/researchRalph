#!/bin/bash
# Harness for imo1993p5 — checks if solution.lean compiles
set -e

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
SOLUTION="$DOMAIN_DIR/solution.lean"

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

# Log to experiments.jsonl
EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 1) )))"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Append to results.tsv
if [ ! -f "$DOMAIN_DIR/results.tsv" ]; then
    printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "$DOMAIN_DIR/results.tsv"
fi
printf "%s\t%s\t%s\t%s\t%s\n" "$EXP_ID" "$SCORE" "$STATUS" "proof attempt" "$AGENT" >> "$DOMAIN_DIR/results.tsv"

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
