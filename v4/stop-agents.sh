#!/bin/bash
# stop-agents.sh — kill all rrma screen sessions

echo "Stopping all RRMA sessions..."
for s in $(screen -ls | grep -oE 'rrma-[a-z0-9]+' | sort -u); do
    screen -S "$s" -X quit 2>/dev/null && echo "  Killed $s"
done
echo "Done."
