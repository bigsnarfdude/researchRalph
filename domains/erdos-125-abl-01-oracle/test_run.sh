#!/bin/bash
set -e

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="${RRMA_LEAN_PROJECT:-$HOME/rrma-lean}"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="$HOME/.elan/bin:$PATH"

echo "DOMAIN_DIR=$DOMAIN_DIR"
echo "LEAN_PROJECT=$LEAN_PROJECT"
echo "AGENT=$AGENT"

WORKSPACE_FILE="$DOMAIN_DIR/workspace/$AGENT/Erdos125.lean"
if [ -f "$WORKSPACE_FILE" ]; then
    SOLUTION="$WORKSPACE_FILE"
else
    SOLUTION="$DOMAIN_DIR/Erdos125.lean"
fi

echo "SOLUTION=$SOLUTION"

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"
    echo "STATUS: Erdos125.lean not found"
    exit 0
fi

echo "File exists, continuing..."

SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" | grep -c "sorry" 2>/dev/null)
SORRY_COUNT=${SORRY_COUNT:-0}

echo "SORRY_COUNT=$SORRY_COUNT"

TMP_FILE="$LEAN_PROJECT/Erdos125Test_${AGENT}_$$.lean"
echo "TMP_FILE=$TMP_FILE"

cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
set +e
echo "About to run lake env lean..."
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
set -e
rm -f "$TMP_FILE"

echo "Rest of script..."
