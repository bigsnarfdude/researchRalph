#!/bin/bash
# Harness: run GPT-2 TinyStories training and log results
# Usage: bash run.sh <exp_name> "description" design_type
#
# v4.7: Agent-local workspaces eliminate race conditions.
# Each agent edits workspace/agentN/train.py — no shared mutable file.
# Serializes GPU access with flock so multiple agents don't OOM.
# Automatically appends score to results.tsv.

METHOD=${1:-baseline}
DESCRIPTION=${2:-"no description"}
DESIGN=${3:-hyperparam}

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
LOGS_DIR="$DOMAIN_DIR/logs"
LOCKFILE="$DOMAIN_DIR/.gpu.lock"

mkdir -p "$LOGS_DIR"

AGENT="${CLAUDE_AGENT_ID:-manual}"
RUN_LOG="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s).log"

# v4.7: Resolve agent-local train.py (workspace/agentN/train.py)
# Falls back to domain-level train.py for manual runs or backwards compat
WORKSPACE="$DOMAIN_DIR/workspace/$AGENT"
if [ -f "$WORKSPACE/train.py" ]; then
    TRAIN_PY="$WORKSPACE/train.py"
elif [ -f "$DOMAIN_DIR/train.py" ]; then
    TRAIN_PY="$DOMAIN_DIR/train.py"
else
    echo "[run.sh] ERROR: No train.py found in workspace/$AGENT/ or domain root"
    exit 1
fi

echo "[run.sh] Queuing experiment: $METHOD (using $TRAIN_PY)"
echo "[run.sh] Results will appear in results.tsv when done (~7 min)."

# Run in background so agents aren't blocked by 2-min bash timeout
(
    flock -x 200

    echo "[run.sh] GPU lock acquired. Starting training..."

    # Snapshot the agent's train.py at flock-acquire time
    SNAPSHOT="$LOGS_DIR/${METHOD}_${AGENT}_$(date +%s)_train.py"
    cp "$TRAIN_PY" "$SNAPSHOT"
    CONFIG_HASH=$(md5sum "$SNAPSHOT" 2>/dev/null | cut -d' ' -f1 || md5 -q "$SNAPSHOT" 2>/dev/null || echo "nohash")

    # Copy agent's train.py to domain root for the actual training run
    # (training scripts expect train.py in the domain directory)
    cp "$TRAIN_PY" "$DOMAIN_DIR/train.py"

    START=$(date +%s)

    # 10-minute timeout (5-min budget + compilation overhead)
    timeout 600 uv run train.py > "$RUN_LOG" 2>&1 || true
    EXIT_CODE=$?

    END=$(date +%s)
    TRAIN_MIN=$(( (END - START) / 60 ))

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[run.sh] CRASH (exit $EXIT_CODE)"
        SCORE="crash"
        VRAM="0"
        STATUS="crash"
    else
        SCORE=$(grep "^val_bpb:" "$RUN_LOG" | tail -1 | awk '{print $2}')
        VRAM=$(grep "^peak_vram_mb:" "$RUN_LOG" | tail -1 | awk '{print $2}')
        STATUS="discard"

        if [ -z "$SCORE" ]; then
            echo "[run.sh] No val_bpb found in output"
            SCORE="crash"
            STATUS="crash"
        fi
    fi

    # Generate EXP-ID
    LAST_N=$(grep -oP 'exp\K\d+' "$RESULTS_TSV" 2>/dev/null | sort -n | tail -1 || echo 0)
    NEXT_N=$(printf "%03d" $((10#${LAST_N:-0} + 1)))
    EXP_ID="exp${NEXT_N}"

    # Determine keep/discard
    if [ "$STATUS" != "crash" ]; then
        CURRENT_BEST=$(awk -F'\t' 'NR>1 && $4=="keep" {print $2}' "$RESULTS_TSV" 2>/dev/null | sort -n | head -1)
        if [ -z "$CURRENT_BEST" ]; then
            STATUS="keep"
        elif python3 -c "exit(0 if float('$SCORE') < float('$CURRENT_BEST') else 1)" 2>/dev/null; then
            STATUS="keep"
        fi
    fi

    # Append to results
    echo -e "${EXP_ID}\t${SCORE}\t${VRAM}\t${STATUS}\t${DESCRIPTION}\t${AGENT}\t${DESIGN}\t${TRAIN_MIN}" >> "$RESULTS_TSV"

    # Update best/ if new best — use the snapshot, not live train.py
    if [ "$STATUS" = "keep" ]; then
        cp "$SNAPSHOT" "$DOMAIN_DIR/best/train.py"
        echo "$CONFIG_HASH" > "$DOMAIN_DIR/best/config_hash"
        echo "[run.sh] NEW BEST: $SCORE — saved to best/train.py (hash: $CONFIG_HASH)"
    fi

    echo ""
    echo "[run.sh] $EXP_ID: val_bpb=$SCORE vram=${VRAM}MB status=$STATUS"
    echo "val_bpb: $SCORE"

) 200>"$LOCKFILE" &
echo "[run.sh] Background PID: $!"
