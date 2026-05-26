#!/bin/bash
set -e
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="/home/vincent/.elan/bin:$PATH"

# ABLATION 02: workspace fix removed — always reads domain root, ignores workspace/
SOLUTION="$DOMAIN_DIR/Erdos125.lean"

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"; echo "STATUS: Erdos125.lean not found"; exit 0
fi

SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" | grep -c "sorry" 2>/dev/null) || true
SORRY_COUNT=${SORRY_COUNT:-0}

TMP_FILE="$LEAN_PROJECT/Erdos125Test.lean"
cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
rm -f "$TMP_FILE"

echo "=== ORACLE ==="
echo "SORRY_COUNT: $SORRY_COUNT"
echo "BUILD_EXIT: $BUILD_EXIT"
echo "SOURCE: $SOLUTION"

MAX_SORRY=4
if [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    echo "SCORE=1.0"; echo "STATUS: PROVED"
    SCORE="1.0"; STATUS="proved"

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
    cp "$SOLUTION" "$DOMAIN_DIR/Erdos125.lean"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: COMPILE_ERROR — $SORRY_COUNT sorry"
    echo "$BUILD_OUT" | tail -30
    SCORE="$FRAC"; STATUS="compile_error"
else
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    SCORE="$FRAC"; STATUS="in_progress"
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"
