#!/bin/bash
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="${RRMA_LEAN_PROJECT:-$HOME/rrma-lean}"
AGENT="${CLAUDE_AGENT_ID:-agent0}"

SOLUTION="$DOMAIN_DIR/workspace/$AGENT/Erdos125.lean"

# Correct approach: grep -c will output the count (0 or more)
# Then use || true to avoid exit code 1 when count is 0
SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" 2>/dev/null | grep -c "sorry" 2>/dev/null || true)
SORRY_COUNT=${SORRY_COUNT:-0}  # fallback to 0 if empty

TMP_FILE="$LEAN_PROJECT/Erdos125Test_${AGENT}_$$.lean"
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
    echo "SCORE=1.0"
    echo "STATUS: PROVED ✓"
else
    echo "SCORE=0.0"
    echo "STATUS: NOT_PROVED"
fi
