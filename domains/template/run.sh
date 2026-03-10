#!/bin/bash
# Harness: run one experiment and output the score
#
# Usage: bash run.sh [config-file]
#
# Must:
# - Accept a config file path
# - Run the experiment
# - Print the score to stdout
# - Exit cleanly (timeout if needed)

CONFIG="${1:-config.yaml}"

# Example: ML training
# timeout 300 python train.py --config "$CONFIG" > run.log 2>&1
# grep "val_score" run.log | tail -1 | awk '{print $NF}'

# Example: Benchmark
# make clean && make CFLAGS="$(cat $CONFIG)" && ./benchmark 2>/dev/null | tail -1

# Example: Prompt eval
# python eval.py --prompt "$CONFIG" 2>/dev/null | grep "accuracy" | awk '{print $NF}'

echo "ERROR: Edit this harness for your domain"
exit 1
