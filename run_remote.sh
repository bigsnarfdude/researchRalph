#!/bin/bash
# Run experiment on nigel via SSH
# Usage: ./run_remote.sh probes/exp001_multimax.py

set -e

PROBE_FILE=$1
REMOTE_HOST="vincent@nigel.birs.ca"
REMOTE_DIR="~/researchRalph"

if [ -z "$PROBE_FILE" ]; then
    echo "Usage: ./run_remote.sh probes/exp001_name.py"
    exit 1
fi

PROBE_NAME=$(basename "$PROBE_FILE")
EXP_ID="${PROBE_NAME%.py}"

echo "═══════════════════════════════════════════════════════════"
echo "  Remote Experiment: $EXP_ID"
echo "  Host: $REMOTE_HOST"
echo "═══════════════════════════════════════════════════════════"

# 1. Copy probe file to nigel
echo "[1/4] Copying probe to nigel..."
scp "$PROBE_FILE" "$REMOTE_HOST:$REMOTE_DIR/probes/"

# 2. Run experiment on nigel
echo "[2/4] Running experiment on nigel (this may take a few minutes)..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && python train_eval.py --probe probes/$PROBE_NAME"

# 3. Copy results back
echo "[3/4] Copying results back..."
scp "$REMOTE_HOST:$REMOTE_DIR/results/$EXP_ID.json" "results/"

# 4. Show results
echo "[4/4] Results:"
cat "results/$EXP_ID.json"

echo ""
echo "Done. Results saved to results/$EXP_ID.json"
