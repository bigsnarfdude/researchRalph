#!/bin/bash
# Stop all researchRalph sessions on this machine
echo "Stopping all ralph screen sessions..."
for s in $(screen -ls 2>/dev/null | grep -oP '\d+\.ralph[^\s]*' || true); do
    echo "  killing $s"
    screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "claude -p" 2>/dev/null || true
pkill -f train.py 2>/dev/null || true
echo "Done."
