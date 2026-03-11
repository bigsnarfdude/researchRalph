#!/bin/bash
# Acrobot Swing-Up Harness — runs one experiment and outputs the score
#
# Usage: bash run.sh [config-file]
# Score: avg(1.0 - steps/500) across 50 episodes (0.0 to ~0.84, higher is better)

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Run 50 episodes and output average score
python3 "$DIR/engine.py" "$CONFIG" --matches 50 --seed 42
