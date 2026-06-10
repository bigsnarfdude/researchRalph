#!/bin/bash
# Oracle for Erdős #741(ii) — G3 scaffold
# Binary score: proof compiles without sorry = 1.0, else 0.0

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="/home/vincent/.elan/bin:$PATH"

WORKSPACE_FILE="$DOMAIN_DIR/workspace/$AGENT/Erdos741OAI.lean"
if [ -f "$WORKSPACE_FILE" ]; then
    SOLUTION="$WORKSPACE_FILE"
else
    SOLUTION="$DOMAIN_DIR/Erdos741OAI_reference.lean"
fi

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"
    echo "STATUS: Erdos741OAI.lean not found at $SOLUTION"
    exit 0
fi

SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" | grep -c "sorry" 2>/dev/null) || true
SORRY_COUNT=${SORRY_COUNT:-0}

TMP_FILE="$LEAN_PROJECT/Erdos741OAITest.lean"
cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
rm -f "$TMP_FILE"

echo "=== ORACLE ==="
echo "SORRY_COUNT: $SORRY_COUNT"
echo "BUILD_EXIT: $BUILD_EXIT"
echo "SOURCE: $SOLUTION"

if [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    SCORE="1.0"; STATUS="proved"
    echo "SCORE=1.0"; echo "STATUS: PROVED"
    cp "$SOLUTION" "$DOMAIN_DIR/Erdos741OAI_proved.lean"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    SCORE="0.0"; STATUS="compile_error"
    echo "SCORE=0.0"; echo "STATUS: COMPILE_ERROR — $SORRY_COUNT sorry"
    echo "$BUILD_OUT" | grep "error:" | head -20
else
    SCORE="0.0"; STATUS="in_progress"
    echo "SCORE=0.0"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
fi

RESULTS="$DOMAIN_DIR/results.tsv"
LOCK="$DOMAIN_DIR/results.lock"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DESCRIPTION="proof attempt $TIMESTAMP"

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
