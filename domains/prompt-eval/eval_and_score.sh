#!/bin/bash
# eval_and_score.sh — prompt evaluation harness
# Generates responses using the prompt config, scores with Claude judge
#
# Usage: bash eval_and_score.sh [prompt_config.yaml]
#
# Outputs: result.json with {mean_score, clarity, accuracy, engagement, combined}
# Requires: ANTHROPIC_API_KEY set (uses claude -p for both generation and judging)

set -e

CONFIG="${1:-prompt_config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAMPLES="samples_${TIMESTAMP}.jsonl"
SCORES="scores_${TIMESTAMP}.json"
RESULT="result.json"

echo "=== Prompt Evaluation Harness ==="
echo "Config: $CONFIG"
echo "Timestamp: $TIMESTAMP"
echo ""

# 1. Generate responses using the prompt config
echo "--- Step 1: Generate responses ---"
python3 generate.py --config "$CONFIG" --output "$SAMPLES"
echo ""

# 2. Score with Claude judge
echo "--- Step 2: Score with Claude judge ---"
python3 score.py --input "$SAMPLES" --output "$SCORES" --scores-log "scores_${TIMESTAMP}.jsonl"
echo ""

# 3. Compute aggregate metrics
echo "--- Step 3: Compute metrics ---"
python3 metrics.py --scores "$SCORES" --output "$RESULT"
echo ""

echo "=== Done ==="
echo "Samples: $SAMPLES"
echo "Scores: $SCORES"
echo "Result: $RESULT"
cat "$RESULT"
