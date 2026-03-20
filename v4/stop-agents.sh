#!/bin/bash
# stop-agents.sh — kill rrma worker and meta sessions (NOT the outer loop)

echo "Stopping RRMA worker + meta sessions..."
for s in $(screen -ls 2>/dev/null | grep -oE '[0-9]+\.rrma-(worker|meta)' | cut -d. -f1); do
    screen -S "$s" -X quit 2>/dev/null && echo "  Killed session $s"
done
echo "Done. (outer loop preserved)"
