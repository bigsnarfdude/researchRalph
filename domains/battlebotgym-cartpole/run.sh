#!/bin/bash
# CartPole Control Harness — runs one experiment and outputs the score
#
# Usage: bash run.sh [config-file]
# Score: average survival fraction (0.0 to 1.0, higher is better)

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Run 50 episodes and output normalized score
python3 "$DIR/engine.py" "$CONFIG" --matches 50 --seed 42
