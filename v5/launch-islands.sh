#!/bin/bash
# launch-islands.sh — v5.1 entry point: islands + one loop.sh per island.
#
# Replaces marathon-session launches (v4/launch-agents.sh) for island runs.
# Loops run under setsid so they survive ssh disconnects; each spawns short
# claude sessions per experiment (see loop.sh header for env knobs).
#
# Usage: bash v5/launch-islands.sh /path/to/base-domain [K]
#   Env passthrough: MODEL MAX_EXPS COST_CAP WALL_CAP_H STAG_N SESSION_TURNS ...
#   RESET=1 recreates islands from scratch (default: reuse existing).

set -u
BASE="${1:?usage: launch-islands.sh /path/to/base-domain [K]}"
K="${2:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$BASE" && pwd)"

if [ "${RESET:-0}" = "1" ]; then
    bash "$SCRIPT_DIR/make-islands.sh" "$BASE" "$K" --force
    for d in "$BASE"-isl-*; do chmod 444 "$d/results.tsv" 2>/dev/null; done
fi

SUFFIXES=(a b c d e f g h)
for i in $(seq 0 $((K-1))); do
    ISL="$BASE-isl-${SUFFIXES[$i]}"
    [ -d "$ISL" ] || { echo "ERROR: $ISL missing (run with RESET=1?)" >&2; exit 1; }
    LOG="$ISL/logs/loop.out"
    mkdir -p "$ISL/logs"
    setsid nohup bash "$SCRIPT_DIR/loop.sh" "$ISL" >> "$LOG" 2>&1 &
    echo "[launch-islands] loop for $(basename "$ISL") -> pid $! (log: $LOG)"
done
echo "[launch-islands] stop with: pkill -f 'v5/loop.s[h]'"
