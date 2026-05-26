#!/bin/bash
set -e
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"

# Workspace isolation: prefer agent workspace file
if [ -n "$CLAUDE_AGENT_ID" ] && [ -f "$DOMAIN_DIR/workspace/$CLAUDE_AGENT_ID/Erdos741ii.lean" ]; then
    SOLUTION="$DOMAIN_DIR/workspace/$CLAUDE_AGENT_ID/Erdos741ii.lean"
else
    SOLUTION="$DOMAIN_DIR/Erdos741ii.lean"
fi
export PATH="/home/vincent/.elan/bin:$PATH"

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"; echo "STATUS: Erdos741ii.lean not found"; exit 0
fi

SORRY_COUNT=$(grep -c 'sorry' "$SOLUTION" 2>/dev/null || echo 0)

TMP_FILE="$LEAN_PROJECT/Erdos741iiTest.lean"
cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
rm -f "$TMP_FILE"

echo '=== ORACLE ==='
echo "SORRY_COUNT: $SORRY_COUNT"
echo "BUILD_EXIT: $BUILD_EXIT"

if [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    SCORE="1.0"; STATUS="PROVED"
    echo "SCORE=1.0"; echo "STATUS: PROVED"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    SCORE="0.0"; STATUS="COMPILE_ERROR"
    echo "SCORE=0.0"; echo "STATUS: COMPILE_ERROR"
    echo "$BUILD_OUT" | tail -30
else
    SCORE="0.0"; STATUS="IN_PROGRESS"
    echo "SCORE=0.0"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    echo "$BUILD_OUT" | grep 'error:\|warning:' | head -20
fi

EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 1) - 1 + 1 )))"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
DESCRIPTION="proof attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# RH Prevention: flock + chmod ensures only oracle writes results.tsv
RESULTS="$DOMAIN_DIR/results.tsv"
LOCK="$DOMAIN_DIR/results.lock"
(
    flock -x -w 30 200 || { echo '[oracle] Could not acquire results.lock'; exit 0; }
    chmod 644 "$RESULTS" 2>/dev/null || true
    if [ ! -f "$RESULTS" ] || ! head -1 "$RESULTS" | grep -q '^EXP-ID'; then
        printf 'EXP-ID\tscore\tstatus\tdescription\tagent\n' > "$RESULTS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$EXP_ID" "$SCORE" "$STATUS" "$DESCRIPTION" "$AGENT" >> "$RESULTS"
    chmod 444 "$RESULTS"
) 200>"$LOCK"
