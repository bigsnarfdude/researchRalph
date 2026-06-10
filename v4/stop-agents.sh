#!/bin/bash
# stop-agents.sh — kill rrma worker and meta sessions (NOT the outer loop)
# Honors RRMA_PREFIX (default "rrma") so concurrent fleets can be stopped independently.

PREFIX="${RRMA_PREFIX:-rrma}"

echo "Stopping ${PREFIX} worker + meta sessions..."
for s in $(screen -ls 2>/dev/null | grep -oE "[0-9]+\.${PREFIX}-(worker[0-9]*|meta)" | cut -d. -f1); do
    screen -S "$s" -X quit 2>/dev/null && echo "  Killed session $s"
done
echo "Done. (outer loop preserved)"
