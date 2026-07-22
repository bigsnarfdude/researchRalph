#!/bin/bash
# cartpole-island oracle — v4.9.3 contract (workspace + SOURCE + results.tsv).
#
# Usage: bash run.sh <name> "description"     (CLAUDE_AGENT_ID set by harness)
# Score: average survival fraction over 50 episodes, fixed seed. Deterministic.

AGENT="${CLAUDE_AGENT_ID:-agent0}"
NAME="${1:-exp}"
DESC="${2:-no description}"
DIR="$(cd "$(dirname "$0")" && pwd)"
WS="$DIR/workspace/$AGENT/params.yaml"

if [ ! -f "$WS" ]; then
    echo "ERROR: $WS missing — edit workspace/$AGENT/params.yaml, not the domain root" >&2
    exit 1
fi

echo "SOURCE: $WS"
OUT=$(python3 "$DIR/engine.py" "$WS" --matches 50 --seed 42 2>&1)
echo "$OUT"
SCORE=$(echo "$OUT" | tail -1)
case "$SCORE" in
    ''|*[!0-9.]*) echo "ERROR: engine did not return a score (got: $SCORE)" >&2; exit 1 ;;
esac

N=$(awk 'NR>1' "$DIR/results.tsv" | wc -l | tr -d ' ')
EXP=$(printf "EXP-%03d" $((N+1)))
chmod 644 "$DIR/results.tsv"
printf "%s\t%s\t0.0\tkeep\t%s\t%s\t%s\n" "$EXP" "$SCORE" "$DESC" "$AGENT" "$NAME" >> "$DIR/results.tsv"
chmod 444 "$DIR/results.tsv"

echo "SCORE: $SCORE"
echo "LOGGED: $EXP -> results.tsv"
