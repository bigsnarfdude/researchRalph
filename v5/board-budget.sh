#!/bin/bash
# board-budget.sh — line-budget check for an island blackboard.
#
# Design decision (v5.0): warn + flag, never reject. A rejected write makes
# agents fail opaquely mid-loop; a flag file is visible to the gardener and
# to agents (curation becomes their job). The board stays in git, so nothing
# is ever lost by curation — only demoted.
#
# Usage: bash board-budget.sh /path/to/island [cap]
#   cap defaults to $BOARD_BUDGET or 300.
# Exit 0 = under budget (stale flag removed). Exit 2 = over budget (flag set).

set -euo pipefail

ISL="${1:?usage: board-budget.sh /path/to/island [cap]}"
CAP="${2:-${BOARD_BUDGET:-300}}"
BOARD="$ISL/blackboard.md"
FLAG="$ISL/BOARD_OVER_BUDGET"

LINES=$(wc -l < "$BOARD" | tr -d ' ')

if [ "$LINES" -gt "$CAP" ]; then
    printf 'over budget: %s lines (cap %s) — curate blackboard.md down before adding findings\nflagged: %s\n' \
        "$LINES" "$CAP" "$(date '+%Y-%m-%d %H:%M:%S')" > "$FLAG"
    echo "[board-budget] WARN: $(basename "$ISL") board is $LINES lines (cap $CAP) — flag set"
    exit 2
fi

rm -f "$FLAG"
echo "[board-budget] ok: $(basename "$ISL") board is $LINES/$CAP lines"
