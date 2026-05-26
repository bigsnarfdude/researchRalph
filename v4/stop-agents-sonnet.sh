#!/bin/bash
echo "Stopping Sonnet worker + meta sessions..."
for s in $(screen -ls 2>/dev/null | grep -oE "[0-9]+\.rrma-sonnet-(w|meta)" | cut -d. -f1); do
    screen -S "$s" -X quit 2>/dev/null && echo "  Killed session $s"
done
echo "Done."
