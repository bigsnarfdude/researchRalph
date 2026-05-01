#!/bin/bash
# Harness: observe one traffic camera frame, extract demand counts
#
# Usage: bash run.sh <obs_name> "description" design_type
# Score: ns_ew_ratio (NS vehicle count / EW vehicle count)

set -euo pipefail

METHOD="${1:-obs_baseline}"
DESCRIPTION="${2:-no description}"
DESIGN="${3:-prompt}"

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
LOGS_DIR="$DOMAIN_DIR/logs"
mkdir -p "$LOGS_DIR"

AGENT="${CLAUDE_AGENT_ID:-manual}"

WORKSPACE="$DOMAIN_DIR/workspace/$AGENT"
if [ -f "$WORKSPACE/config.yaml" ]; then
    CONFIG_FILE="$WORKSPACE/config.yaml"
elif [ -f "$DOMAIN_DIR/config.yaml" ]; then
    CONFIG_FILE="$DOMAIN_DIR/config.yaml"
else
    echo "[run.sh] ERROR: no config.yaml found"; exit 1
fi

RUN_LOG="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s).log"
SNAPSHOT="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s)_config.yaml"
cp "$CONFIG_FILE" "$SNAPSHOT"

echo "[run.sh] $METHOD | agent=$AGENT | config=$CONFIG_FILE"

START=$(date +%s)
timeout 60 python3 "$DOMAIN_DIR/observe.py" "$CONFIG_FILE" > "$RUN_LOG" 2>&1
EXIT_CODE=$?
END=$(date +%s)
ELAPSED=$((END - START))

NS=$(grep "^ns_count:" "$RUN_LOG" | awk '{print $2}' || echo "0")
EW=$(grep "^ew_count:" "$RUN_LOG" | awk '{print $2}' || echo "0")
RATIO=$(grep "^ns_ew_ratio:" "$RUN_LOG" | awk '{print $2}' || echo "0")
TOTAL=$(grep "^total_count:" "$RUN_LOG" | awk '{print $2}' || echo "0")
CONFIDENCE=$(grep "^confidence:" "$RUN_LOG" | awk '{print $2}' || echo "unknown")
SUCCESS=$(grep "^success:" "$RUN_LOG" | awk '{print $2}' || echo "False")
STATUS="discard"

if [ "$EXIT_CODE" -ne 0 ] || [ "$SUCCESS" != "True" ]; then
    RATIO="crash"; NS="0"; EW="0"; TOTAL="0"; STATUS="crash"
fi

LAST_N=$(grep -oE 'obs[0-9]+' "$RESULTS_TSV" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1 || echo 0)
NEXT_N=$(printf "%03d" $((10#${LAST_N:-0} + 1)))
OBS_ID="obs${NEXT_N}"

if [ "$STATUS" != "crash" ]; then
    CURRENT_BEST=$(awk -F'\t' 'NR>1 && $6=="keep" {print $2}' "$RESULTS_TSV" 2>/dev/null | sort -g | tail -1 || true)
    if [ -z "$CURRENT_BEST" ]; then
        STATUS="keep"
    elif python3 -c "exit(0 if float('$RATIO') > float('$CURRENT_BEST') else 1)" 2>/dev/null; then
        STATUS="keep"
    fi
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$OBS_ID" "$RATIO" "$NS" "$EW" "$TOTAL" "$STATUS" \
    "$DESCRIPTION" "$AGENT" "$DESIGN" "$ELAPSED" \
    >> "$RESULTS_TSV"

if [ "$STATUS" = "keep" ]; then
    cp "$SNAPSHOT" "$DOMAIN_DIR/best/config.yaml"
    echo "[run.sh] NEW BEST ratio=$RATIO ns=$NS ew=$EW confidence=$CONFIDENCE"
fi

# Print demand.json path if written
if [ -f "$DOMAIN_DIR/demand.json" ]; then
    echo "[run.sh] demand.json updated → traffic-signal domain can now use real demand"
fi

echo ""
echo "[run.sh] $OBS_ID: ratio=$RATIO ns=$NS ew=$EW total=$TOTAL status=$STATUS (${ELAPSED}s)"
echo "ns_ew_ratio: $RATIO"
echo "ns_count: $NS"
echo "ew_count: $EW"
