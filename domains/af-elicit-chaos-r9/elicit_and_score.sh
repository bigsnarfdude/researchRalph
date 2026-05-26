#!/bin/bash
# elicit_and_score.sh — AF elicitation harness (replaces train.py)
# Generates samples via vLLM, scores via Claude judge, computes metrics
#
# Usage: ./elicit_and_score.sh [prompt_config.yaml]
#
# Outputs: result.json with {elicit_rate, mean_score, diversity, combined}

set -e

CONFIG="${1:-prompt_config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAMPLES="samples_${TIMESTAMP}.jsonl"
SCORES="scores_${TIMESTAMP}.json"
RESULT="result.json"

echo "=== AF Elicitation Harness ==="
echo "Config: $CONFIG"
echo "Timestamp: $TIMESTAMP"
echo ""

# 1. Generate samples using vLLM
echo "--- Step 1: Generate samples ---"
python3 generate.py --config "$CONFIG" --output "$SAMPLES"
echo ""

# 2. Score with Claude judge
echo "--- Step 2: Score with Claude judge ---"
python3 score.py --input "$SAMPLES" --output "$SCORES" --scores-log "scores_${TIMESTAMP}.jsonl"
echo ""

# 3. Compute metrics
echo "--- Step 3: Compute metrics ---"
python3 metrics.py --scores "$SCORES" --output "$RESULT"
echo ""

echo "=== Done ==="
echo "Samples: $SAMPLES"
echo "Scores: $SCORES"
echo "Result: $RESULT"
cat "$RESULT"
