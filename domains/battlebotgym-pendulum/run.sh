#!/bin/bash
# Pendulum Swing-Up & Balance Harness — runs one experiment and outputs the score
#
# Usage: bash run.sh [config-file]
# Score: normalized reward (0.0 to ~0.95, higher is better)

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Run 50 episodes and output average normalized score
python3 "$DIR/engine.py" "$CONFIG" --matches 50 --seed 42
