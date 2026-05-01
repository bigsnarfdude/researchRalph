#!/bin/bash
# Harness: run one traffic signal timing experiment, log avg_delay to results.tsv
#
# Usage: bash run.sh <exp_name> "description" design_type
# Score: avg_delay (lower is better, units = simulation steps)
#
# v4.7 workspace isolation: reads workspace/$AGENT/config.yaml if present

set -euo pipefail

METHOD="${1:-baseline}"
DESCRIPTION="${2:-no description}"
DESIGN="${3:-timing}"

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
LOGS_DIR="$DOMAIN_DIR/logs"
mkdir -p "$LOGS_DIR"

AGENT="${CLAUDE_AGENT_ID:-manual}"

# v4.7: use agent-local config if available
WORKSPACE="$DOMAIN_DIR/workspace/$AGENT"
if [ -f "$WORKSPACE/config.yaml" ]; then
    CONFIG_FILE="$WORKSPACE/config.yaml"
elif [ -f "$DOMAIN_DIR/config.yaml" ]; then
    CONFIG_FILE="$DOMAIN_DIR/config.yaml"
else
    echo "[run.sh] ERROR: no config.yaml found"
    exit 1
fi

RUN_LOG="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s).log"
SNAPSHOT="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s)_config.yaml"
cp "$CONFIG_FILE" "$SNAPSHOT"
CONFIG_HASH=$(md5sum "$SNAPSHOT" 2>/dev/null | cut -d' ' -f1 || md5 -q "$SNAPSHOT" 2>/dev/null || echo "nohash")

echo "[run.sh] $METHOD | agent=$AGENT | config=$CONFIG_FILE"

START=$(date +%s)
timeout 30 python3 "$DOMAIN_DIR/simulate.py" "$CONFIG_FILE" > "$RUN_LOG" 2>&1
EXIT_CODE=$?
END=$(date +%s)
ELAPSED=$((END - START))

# Parse outputs
SCORE=$(grep "^avg_delay:" "$RUN_LOG" | awk '{print $2}' || true)
THROUGHPUT=$(grep "^throughput:" "$RUN_LOG" | awk '{print $2}' || true)
MAX_QUEUE=$(grep "^max_queue:" "$RUN_LOG" | awk '{print $2}' || true)
NS_DELAY=$(grep "^ns_delay:" "$RUN_LOG" | awk '{print $2}' || true)
EW_DELAY=$(grep "^ew_delay:" "$RUN_LOG" | awk '{print $2}' || true)
SUCCESS=$(grep "^success:" "$RUN_LOG" | awk '{print $2}' || true)
STATUS="discard"

if [ "$EXIT_CODE" -ne 0 ] || [ -z "$SCORE" ] || [ "$SUCCESS" != "True" ]; then
    SCORE="crash"
    THROUGHPUT="0"
    MAX_QUEUE="0"
    NS_DELAY="0"
    EW_DELAY="0"
    STATUS="crash"
fi

# Generate EXP-ID
LAST_N=$(grep -oE 'exp[0-9]+' "$RESULTS_TSV" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1 || echo 0)
NEXT_N=$(printf "%03d" $((10#${LAST_N:-0} + 1)))
EXP_ID="exp${NEXT_N}"

# keep if lower avg_delay than current best
if [ "$STATUS" != "crash" ]; then
    CURRENT_BEST=$(awk -F'\t' 'NR>1 && $5=="keep" {print $2}' "$RESULTS_TSV" 2>/dev/null | sort -g | head -1 || true)
    if [ -z "$CURRENT_BEST" ]; then
        STATUS="keep"
    elif python3 -c "exit(0 if float('$SCORE') < float('$CURRENT_BEST') else 1)" 2>/dev/null; then
        STATUS="keep"
    fi
fi

# Append to results
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$EXP_ID" "$SCORE" "$THROUGHPUT" "$MAX_QUEUE" "$STATUS" \
    "$DESCRIPTION" "$AGENT" "$DESIGN" "$ELAPSED" "$NS_DELAY" \
    >> "$RESULTS_TSV"

# Update best/ on improvement
if [ "$STATUS" = "keep" ]; then
    cp "$SNAPSHOT" "$DOMAIN_DIR/best/config.yaml"
    echo "$CONFIG_HASH" > "$DOMAIN_DIR/best/config_hash"
    echo "[run.sh] NEW BEST avg_delay=$SCORE throughput=$THROUGHPUT ns_delay=$NS_DELAY"
fi

echo ""
echo "[run.sh] $EXP_ID: avg_delay=$SCORE throughput=$THROUGHPUT max_queue=$MAX_QUEUE status=$STATUS (${ELAPSED}s)"
echo "avg_delay: $SCORE"
echo "throughput: $THROUGHPUT"
echo "ns_delay: $NS_DELAY"
echo "ew_delay: $EW_DELAY"
