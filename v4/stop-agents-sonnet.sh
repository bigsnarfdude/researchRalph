#!/bin/bash
# Thin wrapper — stops the sonnet fleet via the canonical stop-agents.sh.
# Also sweeps legacy rrma-sonnet-w*/meta session names from pre-wrapper runs.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for s in $(screen -ls 2>/dev/null | grep -oE "[0-9]+\.rrma-sonnet-(w[0-9]*|meta)" | cut -d. -f1); do
    screen -S "$s" -X quit 2>/dev/null && echo "  Killed legacy session $s"
done
RRMA_PREFIX="${RRMA_PREFIX:-rrma-sonnet}" exec bash "$SCRIPT_DIR/stop-agents.sh"
