#!/bin/bash
# Stop all agents for a domain (or all domains)
#
# Usage:
#   ./core/stop.sh                    # stop all ralph agents
#   ./core/stop.sh gpt2-tinystories   # stop agents for specific domain

DOMAIN="${1:-}"

if [ -n "$DOMAIN" ]; then
    PATTERN="ralph-${DOMAIN}"
else
    PATTERN="ralph-"
fi

echo "Stopping agents matching: $PATTERN"

screen -ls 2>/dev/null | grep "$PATTERN" | awk '{print $1}' | while read -r session; do
    name=$(echo "$session" | cut -d. -f2-)
    screen -S "$session" -X quit 2>/dev/null && echo "  Stopped: $name" || echo "  Already stopped: $name"
done

# Also stop conductor agents
if [ -n "$DOMAIN" ]; then
    CPATTERN="conductor-${DOMAIN}"
else
    CPATTERN="conductor-"
fi

screen -ls 2>/dev/null | grep "$CPATTERN" | awk '{print $1}' | while read -r session; do
    name=$(echo "$session" | cut -d. -f2-)
    screen -S "$session" -X quit 2>/dev/null && echo "  Stopped: $name"
done

echo "Done."
