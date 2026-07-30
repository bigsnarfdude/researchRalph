#!/bin/bash
# loop.sh — v5.1 short-session worker lifecycle for one island.
#
# Lesson from the 2026-07-21 sae-island run: marathon worker sessions die
# (turn exhaustion, CLI notification crashes, wait-by-stopping) and burn cost
# on cache re-reads. v5.1 inverts the unit: the OUTER LOOP (this script, plain
# bash) owns the lifecycle; each model session handles exactly ONE experiment
# and exits. The blackboard is the only memory between sessions.
#
# Per cycle:
#   1. collect any pending oracle RESULT (deterministic, free)
#   2. spawn a short session: analyze last outcome -> one hypothesis -> edit
#      workspace -> submit -> note intent on board -> exit
#   3. wait for the training to finish (bash poll, zero tokens)
#   4. tally session cost into the shared ledger; enforce kill criteria
#
# Kill criteria (all owned here, not by prompts): MAX_EXPS, COST_CAP (shared
# ledger across islands), WALL_CAP_H, stagnation -> BOARD_DISTILL via advisor.
#
# Usage: bash v5/loop.sh /path/to/island
# Env:
#   MODEL=claude-sonnet-5      worker model
#   MAX_EXPS=15                full-fidelity experiments before stopping
#   SESSION_TURNS=30           --max-turns per short session
#   COST_CAP=40                dollars, cumulative across the shared ledger
#   WALL_CAP_H=24              wall-clock hours for this loop
#   STAG_N=8                   full-fidelity exps without a new best -> distill
#   ANCHOR=1                   run the untouched seed as the first row (EXP anchor)
#   ORACLE_TIMEOUT=3600        seconds to wait for a training result
#   ADVISOR_MODEL=claude-sonnet-5   model for BOARD_DISTILL authoring
#   ADVISOR_STUB=<file>        canned advisor output (tests); overrides live call
#   WORKER_CMD=<cmd>           override session spawn (tests). Gets env:
#                              ISLAND_DIR, PROMPT_FILE, EXP_N, SESSION_LOG
#   LEDGER=<path>              cost ledger (default REPO/v5/cost_ledger.tsv)

set -u
ISL="${1:?usage: loop.sh /path/to/island}"
ISL="$(cd "$ISL" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-claude-sonnet-5}"
MAX_EXPS="${MAX_EXPS:-15}"
SESSION_TURNS="${SESSION_TURNS:-30}"
COST_CAP="${COST_CAP:-40}"
WALL_CAP_H="${WALL_CAP_H:-24}"
STAG_N="${STAG_N:-8}"
ANCHOR="${ANCHOR:-1}"
ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-3600}"
ADVISOR_MODEL="${ADVISOR_MODEL:-claude-sonnet-5}"
LEDGER="${LEDGER:-$REPO_ROOT/v5/cost_ledger.tsv}"
AGENT="agent0"
WS="$ISL/workspace/$AGENT"
T_START=$(date +%s)

source "$REPO_ROOT/v4/env.sh" 2>/dev/null || true
mkdir -p "$ISL/logs" "$ISL/.agent_prompts" "$WS"
[ -f "$LEDGER" ] || printf 'ts\tisland\texp\tmodel\tin\tout\tcache_r\tcache_w\tusd\n' > "$LEDGER"

log() { echo "[loop $(basename "$ISL") $(date '+%H:%M:%S')] $*"; }

# --- guards (reward-hack / escape containment). GUARD=0 disables (escape hatch).
GUARD="${GUARD:-1}"
GUARD_SH="$SCRIPT_DIR/guard.sh"
guard_halt() { log "STOP: GUARD — $1"; echo "$(date '+%F %T')  $1" > "$ISL/GUARD_HALT"; }

EDITABLE="$(grep '^editable:' "$ISL/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE="${EDITABLE:-sae.py}"
[ -f "$WS/$EDITABLE" ] || { [ -f "$ISL/$EDITABLE" ] && cp "$ISL/$EDITABLE" "$WS/$EDITABLE"; }

data_rows() { awk -F'\t' 'NR>1 && $4!="smoke"' "$ISL/results.tsv" 2>/dev/null | wc -l | tr -d ' '; }

# ---------------------------------------------------------------- ledger
ledger_lock() {
    local waited=0
    while ! mkdir "$LEDGER.lock" 2>/dev/null; do
        sleep 1; waited=$((waited+1)); [ $waited -ge 30 ] && { rm -rf "$LEDGER.lock"; break; }
    done
}
ledger_unlock() { rmdir "$LEDGER.lock" 2>/dev/null; }

tally_session() {  # $1=session jsonl  $2=exp label   -> appends ledger row
    [ -s "$1" ] || return 0
    local ROW
    ROW=$(python3 - "$1" "$MODEL" <<'PY'
import json, sys
f, model = sys.argv[1], sys.argv[2]
tin=tout=cr=cw=0
for line in open(f, errors="replace"):
    try: d=json.loads(line)
    except: continue
    u=(d.get("message") or {}).get("usage") or {}
    tin+=u.get("input_tokens",0); tout+=u.get("output_tokens",0)
    cr+=u.get("cache_read_input_tokens",0); cw+=u.get("cache_creation_input_tokens",0)
m=model.lower()
if "haiku" in m: r=(1,5,0.1,1.25)
elif "opus" in m: r=(15,75,1.5,18.75)
else: r=(3,15,0.3,3.75)
usd=tin/1e6*r[0]+tout/1e6*r[1]+cr/1e6*r[2]+cw/1e6*r[3]
print(f"{tin}\t{tout}\t{cr}\t{cw}\t{usd:.4f}")
PY
) || return 0
    ledger_lock
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$(basename "$ISL")" "$2" "$MODEL" "$ROW" >> "$LEDGER"
    ledger_unlock
}

ledger_total() { awk -F'\t' 'NR>1 {s+=$NF} END {printf "%.2f", s+0}' "$LEDGER" 2>/dev/null; }

# ---------------------------------------------------------------- advisor
advisor_call() {  # $1=mode(distill)  stdout = advisor text
    if [ -n "${ADVISOR_STUB:-}" ]; then cat "$ADVISOR_STUB"; return; fi
    local SUMMARY
    SUMMARY=$(awk -F'\t' 'NR>1 {print $1, $2, $4, $7, "|", substr($5,1,80)}' "$ISL/results.tsv" | tail -n 40)
    claude -p --model "$ADVISOR_MODEL" --max-turns 3 --dangerously-skip-permissions \
"You are the research advisor for an autonomous experiment loop. Distill this island's blackboard into a replacement board of AT MOST 80 lines with three sections: VERIFIED FINDINGS (each with its exact logged score), EXHAUSTED DIRECTIONS (explicitly closed, with evidence), OPEN FRONTIER (what has genuinely not been tried — be specific but do not invent results). Output ONLY the new board content, no preamble.

results.tsv (recent rows: id score status design | description):
$SUMMARY

blackboard.md:
$(cat "$ISL/blackboard.md")" 2>/dev/null
}

board_distill() {
    log "STAGNATION: no new best in last $STAG_N full experiments -> BOARD_DISTILL"
    local TS NEW
    TS=$(date +%s)
    NEW=$(advisor_call distill)
    if [ -z "$NEW" ]; then log "advisor returned empty — keeping board"; return 1; fi
    cp "$ISL/blackboard.md" "$ISL/blackboard.md.pre-distill-$TS"
    { echo "# Blackboard — distilled by advisor $(date '+%F %T') (previous: blackboard.md.pre-distill-$TS)"
      echo "$NEW"; } > "$ISL/blackboard.md"
    echo "$(data_rows)" > "$ISL/.distill_at_rows"
    bash "$SCRIPT_DIR/board-budget.sh" "$ISL" || true
}

stagnant() {  # true if last STAG_N full rows produced no new best
    python3 - "$ISL/results.tsv" "$STAG_N" "$ISL/.distill_at_rows" <<'PY'
import sys
rows=[]
try:
    for i,l in enumerate(open(sys.argv[1])):
        if i==0: continue
        p=l.rstrip("\n").split("\t")
        if len(p)>=4 and p[3]!="smoke":
            try: rows.append(float(p[1]))
            except: pass
except FileNotFoundError: sys.exit(1)
n=int(sys.argv[2])
try: base=int(open(sys.argv[3]).read().strip())
except: base=0
rows=rows[base:] if base<len(rows) else []
if len(rows)<n: sys.exit(1)
best=0.0; last_imp=0
for i,s in enumerate(rows):
    if s>best: best=s; last_imp=i
sys.exit(0 if len(rows)-1-last_imp>=n-1 else 1)
PY
}

# ---------------------------------------------------------------- oracle glue
collect_pending() {  # returns 0 if something was collected (row logged or error surfaced)
    if [ -f "$WS/RESULT" ] || [ -f "$WS/RESULT_ERROR" ]; then
        ( cd "$ISL" && CLAUDE_AGENT_ID=$AGENT ORACLE_WAIT=1 bash run.sh "${PENDING_NAME:-pending}" "${PENDING_DESC:-collected by loop}" ) > "$ISL/logs/collect.out" 2>&1
        return 0
    fi
    return 1
}

wait_oracle() {  # wait for RESULT/RESULT_ERROR after a submission; 1 on timeout/none
    [ -f "$WS/TRAINING.pid" ] || [ -f "$WS/RESULT" ] || [ -f "$WS/RESULT_ERROR" ] || return 1
    local n=0
    while [ ! -f "$WS/RESULT" ] && [ ! -f "$WS/RESULT_ERROR" ]; do
        sleep 20; n=$((n+20))
        if [ $n -ge "$ORACLE_TIMEOUT" ]; then
            log "oracle timeout after ${ORACLE_TIMEOUT}s"
            return 1
        fi
    done
    return 0
}

# ---------------------------------------------------------------- session
build_prompt() {  # $1=exp_n $2=last_outcome_text -> prompt file path
    local PF="$ISL/.agent_prompts/session_exp$1.md"
    local TEMPLATE="$ISL/session_prompt.md"
    [ -f "$TEMPLATE" ] || TEMPLATE="$SCRIPT_DIR/session_default.md"
    {
        sed -e "s|{{AGENT_ID}}|$AGENT|g" -e "s|{{EDITABLE_FILE}}|$EDITABLE|g" \
            -e "s|{{EXP_N}}|$1|g" "$TEMPLATE"
        echo
        echo "## Last outcome"
        echo "$2"
    } > "$PF"
    echo "$PF"
}

run_session() {  # $1=exp_n $2=last_outcome  -> spawns one short session
    local PF SLOG RC
    PF=$(build_prompt "$1" "$2")
    SLOG="$ISL/logs/exp$(printf '%03d' "$1")_session.jsonl"
    log "session exp$1 starting (model=$MODEL, turns<=$SESSION_TURNS)"
    if [ -n "${WORKER_CMD:-}" ]; then
        ISLAND_DIR="$ISL" PROMPT_FILE="$PF" EXP_N="$1" SESSION_LOG="$SLOG" "$WORKER_CMD"
        RC=$?
    else
        ( cd "$ISL" && CLAUDE_AGENT_ID=$AGENT AGENT_ID=$AGENT ORACLE_WAIT=15 \
            claude --output-format stream-json --verbose \
                --dangerously-skip-permissions \
                --model "$MODEL" --max-turns "$SESSION_TURNS" \
                -p "$(cat "$PF")" ) > "$SLOG" 2>&1
        RC=$?
    fi
    tally_session "$SLOG" "exp$1"
    return $RC
}

# ---------------------------------------------------------------- main
log "start: island=$(basename "$ISL") model=$MODEL max_exps=$MAX_EXPS cost_cap=\$$COST_CAP stag_n=$STAG_N"

if [ "$GUARD" = "1" ]; then
    bash "$GUARD_SH" oracle-snapshot "$ISL" 2>&1 | sed 's/^/[loop] /'
fi

if [ "$ANCHOR" = "1" ] && [ "$(data_rows)" -eq 0 ]; then
    log "anchor: scoring untouched seed as first row"
    ( cd "$ISL" && CLAUDE_AGENT_ID=$AGENT ORACLE_WAIT=5 bash run.sh anchor "untouched seed rerun (loop-owned anchor)" ) > "$ISL/logs/anchor.out" 2>&1
    wait_oracle && PENDING_NAME=anchor PENDING_DESC="untouched seed rerun (loop-owned anchor)" collect_pending
    if [ "$(data_rows)" -eq 0 ]; then
        log "WARNING: anchor produced no row — oracle said: $(tail -n 2 "$ISL/logs/anchor.out" | tr '\n' ' ')"
    else
        log "anchor done: $(awk -F'\t' 'NR>1 {print $1, $2}' "$ISL/results.tsv" | tail -n 1)"
    fi
fi

EXP_N=$(( $(data_rows) + 1 ))
LAST_OUTCOME="First experiment of this loop. Read blackboard.md and results.tsv for prior state."

while true; do
    # ---- kill criteria, checked every cycle, deterministically
    ROWS=$(data_rows)
    if [ "$ROWS" -ge "$MAX_EXPS" ]; then log "STOP: MAX_EXPS ($ROWS/$MAX_EXPS)"; break; fi
    SPENT=$(ledger_total)
    if python3 -c "import sys; sys.exit(0 if float('$SPENT') >= float('$COST_CAP') else 1)"; then
        log "STOP: COST_CAP (\$$SPENT >= \$$COST_CAP)"; break
    fi
    if [ $(( $(date +%s) - T_START )) -ge $(( WALL_CAP_H * 3600 )) ]; then
        log "STOP: WALL_CAP (${WALL_CAP_H}h)"; break
    fi
    if [ "$GUARD" = "1" ]; then
        # Fail CLOSED: halt on ANY nonzero, not just 3. A missing snapshot, an
        # unreadable island, a broken guard — all mean "integrity unknown", and
        # continuing would reward a submission we can no longer vouch for.
        bash "$GUARD_SH" oracle-verify "$ISL" >/dev/null 2>"$ISL/logs/guard.err"; GV=$?
        [ "$GV" -ne 0 ] && { guard_halt "oracle integrity unverifiable (rc=$GV): $(tr '\n' ' ' < "$ISL/logs/guard.err")"; break; }
    fi
    if stagnant; then board_distill || true; fi

    # ---- one experiment cycle
    PREV_ROWS=$ROWS
    run_session "$EXP_N" "$LAST_OUTCOME"; SRC=$?

    # scan the session trace BEFORE collecting its result — a session that took
    # out-of-scope actions must not have its submission rewarded.
    SLOG="$ISL/logs/exp$(printf '%03d' "$EXP_N")_session.jsonl"
    if [ "$GUARD" = "1" ] && [ -s "$SLOG" ]; then
        # Fail CLOSED here too: a scanner that errored out has not cleared this
        # session, and an uncleared session must not have its result collected.
        bash "$GUARD_SH" scan-trace "$SLOG" "$ISL" >/dev/null 2>"$ISL/logs/guard.err"; GS=$?
        [ "$GS" -ne 0 ] && { guard_halt "session trace not cleared (rc=$GS): $(tr '\n' ' ' < "$ISL/logs/guard.err")"; break; }
    fi

    if [ -f "$WS/TRAINING.pid" ] || [ -f "$WS/RESULT" ] || [ -f "$WS/RESULT_ERROR" ]; then
        wait_oracle || true
        PENDING_NAME="exp$EXP_N" PENDING_DESC="collected by loop after session exit" collect_pending || true
    fi

    NEW_ROWS=$(data_rows)
    if [ "$NEW_ROWS" -gt "$PREV_ROWS" ]; then
        LAST=$(awk -F'\t' 'NR>1 && $4!="smoke" {r=$0} END {split(r,p,"\t"); printf "%s scored %s (design %s): %s", p[1], p[2], p[7], p[5]}' "$ISL/results.tsv")
        LAST_OUTCOME="Previous experiment result — $LAST. Analyze it against the board's hypothesis before designing the next change."
        log "exp$EXP_N logged: $(awk -F'\t' 'NR>1 {print $1, $2}' "$ISL/results.tsv" | tail -n 1)"
        EXP_N=$((EXP_N + 1))
    elif ls "$WS"/RESULT_ERROR.* > /dev/null 2>&1; then
        ERR=$(cat "$(ls -t "$WS"/RESULT_ERROR.* | head -n 1)" 2>/dev/null | head -n 3)
        LAST_OUTCOME="Previous submission FAILED (no row logged). Oracle error: ${ERR}. Read workspace/$AGENT/train.err, fix the cause, resubmit. Infrastructure errors are never scores."
        log "exp$EXP_N errored (session rc=$SRC) — next session gets diagnose framing"
    else
        LAST_OUTCOME="Previous session exited (rc=$SRC) WITHOUT submitting an experiment. The workspace may contain uncommitted edits — reconcile them with the board's last stated intent, then submit."
        log "exp$EXP_N: session made no submission (rc=$SRC)"
    fi
done

log "done: rows=$(data_rows) spent_total=\$$(ledger_total) wall=$(( ($(date +%s) - T_START) / 60 ))min"
