#!/bin/bash
# mock-session.sh — stands in for a claude short session in loop.sh tests.
# Env from loop.sh: ISLAND_DIR, PROMPT_FILE, EXP_N, SESSION_LOG.
# MOCK_PLAN="improve flat nosubmit ..." — word k drives call k (last word repeats).
#   improve  : grow the editable file -> rising oracle score
#   flat     : constant content -> constant score (stagnation trigger)
#   nosubmit : write a board note, no oracle call, exit 1 (dead-session sim)
set -eu
CALLS_F="$ISLAND_DIR/.mock_calls"
C=$(cat "$CALLS_F" 2>/dev/null || echo 0); C=$((C+1)); echo "$C" > "$CALLS_F"
PLAN=(${MOCK_PLAN:-improve})
IDX=$((C-1)); [ "$IDX" -ge "${#PLAN[@]}" ] && IDX=$(( ${#PLAN[@]} - 1 ))
MODE="${PLAN[$IDX]}"

EDITABLE="$(grep '^editable:' "$ISLAND_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE="${EDITABLE:-answer.txt}"
WS="$ISLAND_DIR/workspace/agent0"; mkdir -p "$WS"

case "$MODE" in
    nosubmit)
        echo "- [mock exp$EXP_N] session died before submitting" >> "$ISLAND_DIR/blackboard.md"
        printf '{"message":{"usage":{"input_tokens":500,"output_tokens":200}}}\n' > "$SESSION_LOG"
        exit 1 ;;
    flat)
        printf 'constant answer twelve words here to keep the score identical always over runs\n' > "$WS/$EDITABLE" ;;
    improve)
        python3 -c "print('word ' * (10 + $C * 7))" > "$WS/$EDITABLE" ;;
    *)  echo "unknown MOCK_PLAN mode: $MODE" >&2; exit 2 ;;
esac

echo "- [mock exp$EXP_N mode=$MODE] hypothesis: content change moves the score" >> "$ISLAND_DIR/blackboard.md"
( cd "$ISLAND_DIR" && CLAUDE_AGENT_ID=agent0 bash run.sh "mock-exp$EXP_N" "mock $MODE experiment" ) > /dev/null 2>&1 || true
printf '{"message":{"usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":100000,"cache_creation_input_tokens":50000}}}\n' > "$SESSION_LOG"
