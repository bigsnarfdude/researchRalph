#!/bin/bash
# run-kviolation-sweep.sh — bounded-parallelism sweep driver for the
# K-violation 2x2 (v5-hack-gates).
#
# Discovers every domains/<prefix>-* cell dir, launches each through
# v4/launch-kviolation.sh (which itself writes the manifest, snapshots the
# oracle, and injects the framing/salience factors), and keeps at most
# `parallel` cells in flight at once so a laptop/box doesn't get N*2 claude
# processes all at once. Polls screen for liveness; a cell counts as done
# once BOTH its worker screens are gone. A hard wall-clock cap kills and
# logs a wedged cell rather than letting it stall the whole sweep.
#
# Usage: bash v4/run-kviolation-sweep.sh <prefix> [parallel] [max_turns] [model]
#   parallel   defaults to 4
#   max_turns, model — passed straight through to launch-kviolation.sh
#     (which has its own defaults: 60, claude-haiku-4-5-20251001)
#
# Env:
#   WALL_CAP_MIN   default 25 — minutes a cell's screens may live before
#                  being killed and logged as TIMEOUT
#   POLL_SEC       default 15 — polling interval
#   RRMA_PREFIX    default kv — must match what launch-kviolation.sh used,
#                  since session names are derived the same way here
#   DRY_RUN=1      passthrough to launch-kviolation.sh — no screens are
#                  ever started, so every cell reads as "done" on the very
#                  next liveness check (fast, token-free sweep-logic test)

set -u

PREFIX="${1:-}"
PARALLEL="${2:-4}"
MAX_TURNS="${3:-60}"
MODEL="${4:-claude-haiku-4-5-20251001}"

if [ -z "$PREFIX" ]; then
    echo "usage: bash v4/run-kviolation-sweep.sh <prefix> [parallel] [max_turns] [model]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOMAINS_DIR="$REPO_ROOT/domains"
LOG_FILE="$REPO_ROOT/kviolation-sweep.log"

WALL_CAP_MIN="${WALL_CAP_MIN:-25}"
POLL_SEC="${POLL_SEC:-15}"
RRMA_PREFIX="${RRMA_PREFIX:-kv}"
export DRY_RUN="${DRY_RUN:-0}"
export RRMA_PREFIX

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# Portable array population (no mapfile/readarray — this repo targets
# bash 3.2, macOS's default). Domain dir names are assumed space-free, same
# assumption every other v4/v5 script here makes.
ALL_CELLS=($(find "$DOMAINS_DIR" -maxdepth 1 -type d -name "${PREFIX}-*" | sort))
if [ "${#ALL_CELLS[@]}" -eq 0 ]; then
    echo "[run-kviolation-sweep] STOP: no domains matching domains/${PREFIX}-* found" >&2
    exit 1
fi

log "SWEEP_START prefix=$PREFIX cells=${#ALL_CELLS[@]} parallel=$PARALLEL max_turns=$MAX_TURNS model=$MODEL wall_cap_min=$WALL_CAP_MIN dry_run=$DRY_RUN"

SWEEP_START_EPOCH=$(date +%s)

PENDING=("${ALL_CELLS[@]}")
# INFLIGHT / INFLIGHT_START are parallel arrays (index i of one corresponds
# to index i of the other) — bash 3.2 has no associative arrays, so this
# stands in for a cell_dir -> start_epoch map.
INFLIGHT=()
INFLIGHT_START=()
COMPLETED=()
TIMED_OUT=()
FAILED=()

is_alive() {  # $1 = session name substring; true if screen knows about it
    screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

while [ "${#PENDING[@]}" -gt 0 ] || [ "${#INFLIGHT[@]}" -gt 0 ]; do

    # --- fill available slots ---
    while [ "${#INFLIGHT[@]}" -lt "$PARALLEL" ] && [ "${#PENDING[@]}" -gt 0 ]; do
        CELL_DIR="${PENDING[0]}"
        PENDING=("${PENDING[@]:1}")
        CELL_NAME="$(basename "$CELL_DIR")"

        log "LAUNCHING $CELL_NAME"
        if bash "$SCRIPT_DIR/launch-kviolation.sh" "$CELL_DIR" "$MAX_TURNS" "$MODEL" >>"$LOG_FILE" 2>&1; then
            INFLIGHT+=("$CELL_DIR")
            INFLIGHT_START+=("$(date +%s)")
            log "LAUNCHED $CELL_NAME"
        else
            log "LAUNCH_FAILED $CELL_NAME (see $LOG_FILE for launch-kviolation.sh output)"
        fi
    done

    # --- poll in-flight cells ---
    # C-style index loop, not `for idx in "${!INFLIGHT[@]}"` — bash 3.2
    # raises "unbound variable" under `set -u` when a *genuinely empty*
    # array is expanded with [@]/[*], which INFLIGHT legitimately can be
    # (e.g. every launch this round failed). ${#INFLIGHT[@]} is safe (0)
    # even then, so indexing off the count sidesteps the whole class of bug.
    STILL_INFLIGHT=()
    STILL_START=()
    N_INFLIGHT="${#INFLIGHT[@]}"
    idx=0
    while [ "$idx" -lt "$N_INFLIGHT" ]; do
        CELL_DIR="${INFLIGHT[$idx]}"
        START="${INFLIGHT_START[$idx]}"
        idx=$((idx + 1))
        CELL_NAME="$(basename "$CELL_DIR")"
        CELLREP="${CELL_NAME#${PREFIX}-}"
        S0="${RRMA_PREFIX}-${CELLREP}-worker0"
        S1="${RRMA_PREFIX}-${CELLREP}-worker1"
        NOW=$(date +%s)
        ELAPSED=$(( NOW - START ))

        ALIVE=0
        is_alive "$S0" && ALIVE=1
        is_alive "$S1" && ALIVE=1

        if [ "$ALIVE" -eq 0 ]; then
            # Screens gone is NOT the same as work done. The 2026-07-29 sweep
            # reported "40 completed, 0 timed out" while 36 cells had died on an
            # API 429 after ~1 turn — a dead worker and a finished worker look
            # identical from the outside. Classify on evidence instead: oracle
            # rows produced, plus the session's own terminal_reason.
            ROWS=$(( $(wc -l < "$CELL_DIR/results.tsv" 2>/dev/null || echo 1) - 1 ))
            APIERR=""
            for J in "$CELL_DIR"/logs/agent0.jsonl "$CELL_DIR"/logs/agent1.jsonl; do
                [ -f "$J" ] || continue
                if tail -c 4000 "$J" 2>/dev/null | grep -q '"terminal_reason":"api_error"'; then
                    APIERR="$(tail -c 4000 "$J" | grep -o '"api_error_status":[0-9]*' | tail -1)"
                    break
                fi
            done
            if [ -n "$APIERR" ]; then
                FAILED+=("$CELL_DIR")
                log "FAILED $CELL_NAME elapsed_s=$ELAPSED rows=$ROWS reason=api_error ${APIERR:-} — worker died, cell is NOT complete"
            elif [ "$ROWS" -le 0 ]; then
                FAILED+=("$CELL_DIR")
                log "FAILED $CELL_NAME elapsed_s=$ELAPSED rows=0 — workers exited without logging a single oracle call"
            else
                COMPLETED+=("$CELL_DIR")
                log "DONE $CELL_NAME elapsed_s=$ELAPSED rows=$ROWS"
            fi
        elif [ "$ELAPSED" -ge $(( WALL_CAP_MIN * 60 )) ]; then
            screen -S "$S0" -X quit 2>/dev/null
            screen -S "$S1" -X quit 2>/dev/null
            TIMED_OUT+=("$CELL_DIR")
            log "TIMEOUT $CELL_NAME elapsed_s=$ELAPSED wall_cap_min=$WALL_CAP_MIN (screens killed: $S0 $S1)"
        else
            STILL_INFLIGHT+=("$CELL_DIR")
            STILL_START+=("$START")
        fi
    done
    # Same empty-array trap applies to the reassignment itself.
    if [ "${#STILL_INFLIGHT[@]}" -gt 0 ]; then
        INFLIGHT=("${STILL_INFLIGHT[@]}")
        INFLIGHT_START=("${STILL_START[@]}")
    else
        INFLIGHT=()
        INFLIGHT_START=()
    fi

    if [ "${#INFLIGHT[@]}" -gt 0 ] || [ "${#PENDING[@]}" -gt 0 ]; then
        sleep "$POLL_SEC"
    fi
done

SWEEP_END_EPOCH=$(date +%s)
TOTAL_WALL_S=$(( SWEEP_END_EPOCH - SWEEP_START_EPOCH ))

log "SWEEP_END completed=${#COMPLETED[@]} timed_out=${#TIMED_OUT[@]} failed=${#FAILED[@]} total_wall_s=$TOTAL_WALL_S"

echo ""
echo "=== K-VIOLATION SWEEP SUMMARY ==="
echo "Prefix:        $PREFIX"
echo "Cells total:   ${#ALL_CELLS[@]}"
echo "Completed:     ${#COMPLETED[@]}"
echo "Timed out:     ${#TIMED_OUT[@]}"
echo "FAILED:        ${#FAILED[@]}"
if [ "${#TIMED_OUT[@]}" -gt 0 ]; then
    for c in "${TIMED_OUT[@]}"; do echo "  - $(basename "$c")"; done
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  failed cells (re-run these; they produced no usable data):"
    for c in "${FAILED[@]}"; do echo "  - $(basename "$c")"; done
fi
echo "Total wall:    ${TOTAL_WALL_S}s"
echo "Log:           $LOG_FILE"
echo ""
echo "Next: python3 tools/analyze_kviolation.py $PREFIX"
