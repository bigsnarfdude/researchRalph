#!/bin/bash
# Network Routing Optimizer Harness — runs one experiment and outputs the score
#
# Usage: bash run.sh [config-file]
# Score: delivery rate (0.0 to 1.0, higher is better)

CONFIG="${1:-config.yaml}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Run 50 simulations and output average delivery rate
python3 "$DIR/engine.py" "$CONFIG" --matches 50 --seed 42
