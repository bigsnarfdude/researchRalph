#!/bin/bash
# Oracle for Erdős #125 — binary: proof compiles without sorry or it doesn't

set -e

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="/home/vincent/.elan/bin:$PATH"

# Accept workspace file from agent — mirrors ML harness pattern
# Agents edit workspace/$AGENT/Erdos125.lean; run.sh picks it up
WORKSPACE_FILE="$DOMAIN_DIR/workspace/$AGENT/Erdos125.lean"
if [ -f "$WORKSPACE_FILE" ]; then
    SOLUTION="$WORKSPACE_FILE"
else
    SOLUTION="$DOMAIN_DIR/Erdos125.lean"
fi

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"
    echo "STATUS: Erdos125.lean not found"
    exit 0
fi

# Count sorries (exclude comment lines)
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

# Fractional score: (max_sorry - current) / max_sorry gives gradient signal
# Max sorry = 4 (seed state). Score rises as sorries are eliminated.
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
    # Copy winning proof to domain root
    cp "$SOLUTION" "$DOMAIN_DIR/Erdos125.lean"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: COMPILE_ERROR — $SORRY_COUNT sorry, build failed"
    echo "$BUILD_OUT" | tail -30
    SCORE="$FRAC"; STATUS="compile_error"
else
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    echo "$BUILD_OUT" | grep "error:\|warning:\|sorry" | head -20
    SCORE="$FRAC"; STATUS="in_progress"
fi

fi
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"
