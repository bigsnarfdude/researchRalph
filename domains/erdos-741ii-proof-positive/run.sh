#!/bin/bash
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="/home/vincent/.elan/bin:$PATH"

WORKSPACE_FILE="$DOMAIN_DIR/workspace/$AGENT/Erdos741ii.lean"
if [ -f "$WORKSPACE_FILE" ]; then
    SOLUTION="$WORKSPACE_FILE"
else
    SOLUTION="$DOMAIN_DIR/Erdos741ii.lean"
fi

SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" 2>/dev/null | grep -c "sorry" 2>/dev/null)
SORRY_COUNT=${SORRY_COUNT:-0}

TMP_FILE="$LEAN_PROJECT/Erdos741iiTest.lean"
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
    echo "SCORE=1.0"; echo "STATUS: PROVED"
    SCORE="1.0"; STATUS="proved"
    cp "$SOLUTION" "$DOMAIN_DIR/Erdos741ii.lean"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    echo "SCORE=0.0"; echo "STATUS: COMPILE_ERROR"
    echo "$BUILD_OUT" | tail -30
    SCORE="0.0"; STATUS="compile_error"
else
    echo "SCORE=0.5"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    SCORE="0.5"; STATUS="in_progress"
fi

RESULTS="$DOMAIN_DIR/results.tsv"
LOCK="$DOMAIN_DIR/results.lock"
(
    flock -x -w 30 200 || { echo "[oracle] Could not acquire lock"; exit 0; }
    chmod 644 "$RESULTS" 2>/dev/null || true
    if [ ! -f "$RESULTS" ] || ! head -1 "$RESULTS" | grep -q "^EXP-ID"; then
        printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "$RESULTS"
    fi
    EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$RESULTS") )))"
    printf "%s\t%s\t%s\t%s\t%s\n" "$EXP_ID" "$SCORE" "$STATUS" "proof attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)" "$AGENT" >> "$RESULTS"
    chmod 444 "$RESULTS"
) 200>"$LOCK"
