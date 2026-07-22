#!/bin/bash
# island-mock oracle — instant deterministic scorer for island-harness preflight.
#
# Follows the full v4.9.3 oracle contract (the three erdos-125 checks):
#   1. reads the agent workspace file via CLAUDE_AGENT_ID, never the domain root
#   2. echoes "SOURCE: .../workspace/<agent>/..." so preflight can verify #1
#   3. appends an island-mock row to results.tsv
#
# Usage: CLAUDE_AGENT_ID=agent0 bash run.sh [name] ["description"]
# Score: word count of the workspace answer file, capped at 1.0. No model, no GPU.

AGENT="${CLAUDE_AGENT_ID:-agent0}"
NAME="${1:-mock}"
DESC="${2:-mock experiment}"
DIR="$(cd "$(dirname "$0")" && pwd)"
WS="$DIR/workspace/$AGENT/answer.txt"

if [ ! -f "$WS" ]; then
    echo "ERROR: $WS missing — agent workspace not seeded" >&2
    exit 1
fi

echo "SOURCE: $WS"
WORDS=$(wc -w < "$WS" | tr -d ' ')
SCORE=$(python3 -c "print(round(min(1.0, $WORDS/100), 4))")

N=$(awk 'NR>1' "$DIR/results.tsv" | wc -l | tr -d ' ')
EXP=$(printf "EXP-%03d" $((N+1)))
printf "%s\t%s\t0.0\tkeep\t%s\t%s\t%s\n" "$EXP" "$SCORE" "$DESC" "$AGENT" "$NAME" >> "$DIR/results.tsv"

echo "SCORE: $SCORE"
echo "LOGGED: $EXP -> results.tsv"
