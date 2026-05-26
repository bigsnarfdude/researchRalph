#!/bin/bash
set -e
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="/home/vincent/miniF2F-lean4"
SOLUTION="$DOMAIN_DIR/Erdos741ii.lean"
export PATH="/home/vincent/.elan/bin:$PATH"

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"; echo "STATUS: Erdos741ii.lean not found"; exit 0
fi

SORRY_COUNT=$(grep -c "sorry" "$SOLUTION" 2>/dev/null || echo 0)

TMP_FILE="$LEAN_PROJECT/Erdos741iiTest.lean"
cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
rm -f "$TMP_FILE"

echo "=== ORACLE ==="
echo "SORRY_COUNT: $SORRY_COUNT"
echo "BUILD_EXIT: $BUILD_EXIT"

if [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    echo "SCORE=1.0"; echo "STATUS: PROVED"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    echo "SCORE=0.0"; echo "STATUS: COMPILE_ERROR"
    echo "$BUILD_OUT" | tail -30
else
    echo "SCORE=0.0"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    echo "$BUILD_OUT" | grep "error:\|warning:" | head -20
fi

if [ ! -f "$DOMAIN_DIR/results.tsv" ]; then
    printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "$DOMAIN_DIR/results.tsv"
fi
SCORE=$([ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ] && echo "1.0" || echo "0.0")
EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$DOMAIN_DIR/results.tsv") )))"
printf "%s\t%s\t%s\t%s\t%s\n" "$EXP_ID" "$SCORE" "attempt" "$(date -u +%H:%M:%S)" "${CLAUDE_AGENT_ID:-agent0}" >> "$DOMAIN_DIR/results.tsv"
