#!/bin/bash
# Harness: evaluate sorry-elimination progress on DeGiorgi formalization
# Usage: bash run.sh <method_name> "description" design_type

set -e

METHOD=${1:-degiorgi_v1}
DESCRIPTION=${2:-"no description"}
DESIGN=${3:-tactic}

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE=${DEGIORGI_WORKSPACE:-$HOME/rrma-degiorgi-sonnet}
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
LOGS_DIR="$DOMAIN_DIR/logs"

source ~/.elan/env 2>/dev/null || true

mkdir -p "$LOGS_DIR"

# Count total sorries in skeleton (cached after first run)
SORRY_CACHE="$DOMAIN_DIR/.total_sorries"
if [ ! -f "$SORRY_CACHE" ]; then
    TOTAL_SORRIES=$(grep -r "by sorry" "$WORKSPACE/DeGiorgi/" 2>/dev/null | wc -l | tr -d ' ')
    echo "$TOTAL_SORRIES" > "$SORRY_CACHE"
else
    TOTAL_SORRIES=$(cat "$SORRY_CACHE")
fi

# Count remaining sorries
REMAINING=$(grep -r "by sorry" "$WORKSPACE/DeGiorgi/" 2>/dev/null | wc -l | tr -d ' ')

# Try to build
BUILD_LOG="$LOGS_DIR/${METHOD}_$(date +%Y%m%d_%H%M%S).log"
echo "[run.sh] Building DeGiorgi workspace..."
cd "$WORKSPACE"

BUILD_OK=0
if timeout 600 lake build DeGiorgi > "$BUILD_LOG" 2>&1; then
    BUILD_OK=1
    echo "[run.sh] BUILD SUCCESS"
else
    echo "[run.sh] BUILD FAILED (see $BUILD_LOG)"
fi

# Compute score
if [ "$TOTAL_SORRIES" -gt 0 ]; then
    SOLVED=$((TOTAL_SORRIES - REMAINING))
    SCORE=$(python3 -c "print(f'{$SOLVED / $TOTAL_SORRIES:.4f}')")
else
    SCORE="0.0000"
    SOLVED=0
fi

echo "[run.sh] Score: $SCORE ($SOLVED/$TOTAL_SORRIES sorries eliminated, $REMAINING remaining)"
echo "[run.sh] Build: $([ $BUILD_OK -eq 1 ] && echo 'PASS' || echo 'FAIL')"

# Per-module breakdown
echo ""
echo "Module breakdown:"
for dir in Common Foundations EllipticCoefficients LpFunctionToolkit \
           SobolevSpace WeakFormulation Poincare SobolevPoincare \
           PositivePart StampacchiaTruncation BallScaling BallExtension \
           Localization DeGiorgiIteration MoserIteration Supersolutions \
           Crossover Oscillation WeakHarnack Harnack Holder; do
    count=$(grep -r "by sorry" "$WORKSPACE/DeGiorgi/$dir"* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "  $dir: $count sorries remaining"
    else
        echo "  $dir: CLEAN"
    fi
done

# Append to results
AGENT="${CLAUDE_AGENT_ID:-manual}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ ! -f "$RESULTS_TSV" ]; then
    echo -e "EXP-ID\tscore\tsorries_remaining\tbuild_ok\tstatus\tdescription\tagent\tdesign\ttimestamp" > "$RESULTS_TSV"
fi
echo -e "$METHOD\t$SCORE\t$REMAINING\t$BUILD_OK\tkeep\t$DESCRIPTION\t$AGENT\t$DESIGN\t$TIMESTAMP" >> "$RESULTS_TSV"

echo ""
echo "Result appended to $RESULTS_TSV"
