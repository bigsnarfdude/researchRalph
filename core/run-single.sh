#!/bin/bash
# researchRalph v2 — Single-agent loop (simplest mode)
#
# Each iteration gets fresh context, reads state files, runs one experiment.
#
# Usage:
#   ./core/run-single.sh <domain-dir>
#   nohup ./core/run-single.sh domains/gpt2-tinystories > loop.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/run-single.sh <domain-dir>}"

# Resolve to absolute path
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"

# Validate
if [ ! -f "$DOMAIN_DIR/program.md" ]; then
    echo "ERROR: Missing $DOMAIN_DIR/program.md"
    exit 1
fi

# Initialize state files if needed
mkdir -p "$DOMAIN_DIR/best"
[ -f "$DOMAIN_DIR/results.tsv" ] || printf 'commit\tscore\tmemory_gb\tstatus\tdescription\n' > "$DOMAIN_DIR/results.tsv"
[ -f "$DOMAIN_DIR/progress.md" ] || echo "# Progress\n\n## Current Best\n(not yet measured)\n\n## History\n(none yet)" > "$DOMAIN_DIR/progress.md"
[ -f "$DOMAIN_DIR/next_ideas.md" ] || echo "# Experiment Queue\n\n1. Run baseline\n2. (to be filled after baseline)" > "$DOMAIN_DIR/next_ideas.md"

cd "$REPO_DIR"

PROMPT="You are an autonomous AI researcher. Read these files in order:
1. $DOMAIN_NAME/program.md (your full instructions)
2. $DOMAIN_NAME/progress.md (current state and history)
3. $DOMAIN_NAME/next_ideas.md (experiment queue)
4. $DOMAIN_NAME/results.tsv (full results log)

Then run exactly ONE experiment from the top of next_ideas.md.
After the experiment, update all state files:
- Append result to results.tsv
- Update progress.md with result and insights
- Update next_ideas.md: remove tried idea, re-rank, add new ideas

IMPORTANT: Do exactly one experiment then stop. The loop wrapper will call you again."

iteration=0
while true; do
    iteration=$((iteration + 1))
    echo "=========================================="
    echo "RALPH LOOP ITERATION $iteration"
    echo "TIME: $(date)"
    echo "=========================================="

    claude -p "$PROMPT" \
        --dangerously-skip-permissions \
        --max-turns 50

    exit_code=$?
    echo "Claude exited with code: $exit_code"
    echo "Completed iteration $iteration at $(date)"

    sleep 5
done
