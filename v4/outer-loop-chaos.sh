#!/bin/bash
# outer-loop-chaos.sh — runs the chaos experiment using the patched launcher
# Usage: bash outer-loop-chaos.sh /path/to/domain [max_generations] [num_agents] [max_turns] [monitor_interval_min]
#
# Identical to outer-loop.sh but substitutes launch-agents-chaos.sh

DOMAIN_DIR="${1:-.}"
MAX_GEN="${2:-3}"
NUM_AGENTS="${3:-2}"
MAX_TURNS="${4:-50}"
MONITOR="${5:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Temporarily swap the launcher
ORIG="$SCRIPT_DIR/launch-agents.sh"
CHAOS="$SCRIPT_DIR/launch-agents-chaos.sh"
BACKUP="$SCRIPT_DIR/launch-agents.sh.bak"

echo "=== CHAOS EXPERIMENT OUTER LOOP ==="
echo "Swapping launcher: launch-agents.sh → launch-agents-chaos.sh"

cp "$ORIG" "$BACKUP"
cp "$CHAOS" "$ORIG"

# Run the normal outer loop (which will call our chaos launcher)
bash "$SCRIPT_DIR/outer-loop.sh" "$DOMAIN_DIR" "$MAX_GEN" "$NUM_AGENTS" "$MAX_TURNS" "$MONITOR"
EXIT_CODE=$?

# Restore original launcher
echo "Restoring original launch-agents.sh"
mv "$BACKUP" "$ORIG"

exit $EXIT_CODE
