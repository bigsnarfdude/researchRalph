#!/bin/bash
# loop-preflight.sh — v5.1 lifecycle test suite. $0 of model spend.
# Tests loop.sh end-to-end against the island-mock domain with mock sessions.
# Usage: bash v5/loop-preflight.sh    Exit 0 = all pass.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BASE="$REPO_ROOT/domains/island-mock"
ISL="$REPO_ROOT/domains/island-mock-isl-a"
TLEDGER="$SCRIPT_DIR/.test_ledger.tsv"

PASS=0; FAIL=0; declare -a FAILURES=()
ok()  { echo "  ok: $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); FAILURES+=("$1"); }
hdr() { echo; echo "== $1"; }

reset_island() {
    bash "$SCRIPT_DIR/make-islands.sh" "$BASE" 1 --force > /dev/null
    rm -f "$TLEDGER" "$ISL/.mock_calls"
}
rows() { awk -F'\t' 'NR>1 && $4!="smoke"' "$ISL/results.tsv" 2>/dev/null | wc -l | tr -d ' '; }

LOOP_ENV="WORKER_CMD=$SCRIPT_DIR/mock-session.sh LEDGER=$TLEDGER MODEL=claude-sonnet-5 ORACLE_TIMEOUT=60"

# ---------------------------------------------------------------- L1
hdr "L1: anchor + improve cycle + MAX_EXPS stop + ledger"
reset_island
OUT=$(env $LOOP_ENV MOCK_PLAN="improve" ANCHOR=1 MAX_EXPS=3 STAG_N=99 bash "$SCRIPT_DIR/loop.sh" "$ISL" 2>&1)
R=$(rows)
[ "$R" -eq 3 ] && ok "stopped at exactly MAX_EXPS=3 rows" || bad "rows=$R (expected 3)"
echo "$OUT" | grep -q "STOP: MAX_EXPS" && ok "stop reason logged" || bad "no MAX_EXPS stop line"
A=$(awk -F'\t' 'NR==2 {print $7}' "$ISL/results.tsv")
[ "$A" = "anchor" ] && ok "first row is the loop-owned anchor" || bad "first row design='$A' (expected anchor)"
NL=$(awk 'NR>1' "$TLEDGER" 2>/dev/null | wc -l | tr -d ' ')
[ "$NL" -ge 2 ] && ok "ledger has $NL session rows" || bad "ledger rows=$NL (expected >=2)"
USD=$(awk -F'\t' 'NR>1 {s+=$NF} END {printf "%.4f", s+0}' "$TLEDGER")
python3 -c "import sys; sys.exit(0 if float('$USD') > 0 else 1)" && ok "ledger cost accumulates (\$$USD)" || bad "ledger cost is zero"
grep -q "scored" "$ISL/.agent_prompts/session_exp3.md" 2>/dev/null \
    && ok "later session prompt carries previous score" || bad "exp3 prompt lacks previous-score framing"

# ---------------------------------------------------------------- L2
hdr "L2: stagnation -> BOARD_DISTILL (canned advisor), then continues to MAX_EXPS"
reset_island
OUT=$(env $LOOP_ENV MOCK_PLAN="flat" ANCHOR=0 MAX_EXPS=6 STAG_N=3 \
      ADVISOR_STUB="$SCRIPT_DIR/fixtures/canned_distill.md" bash "$SCRIPT_DIR/loop.sh" "$ISL" 2>&1)
head -n 1 "$ISL/blackboard.md" | grep -q "distilled by advisor" && ok "board replaced by distill" || bad "board not distilled"
ls "$ISL"/blackboard.md.pre-distill-* > /dev/null 2>&1 && ok "pre-distill board preserved" || bad "no pre-distill backup"
grep -q "OPEN FRONTIER" "$ISL/blackboard.md" && ok "distilled content in place" || bad "distill content missing"
N_DISTILL=$(echo "$OUT" | grep -c "BOARD_DISTILL")
[ "$N_DISTILL" -eq 1 ] && ok "exactly one distill fired" || bad "distill fired $N_DISTILL times (expected 1)"
[ "$(rows)" -eq 6 ] && ok "loop continued to MAX_EXPS after distill" || bad "rows=$(rows) (expected 6)"

# ---------------------------------------------------------------- L3
hdr "L3: COST_CAP stops the loop before any session"
reset_island
printf 'ts\tisland\texp\tmodel\tin\tout\tcache_r\tcache_w\tusd\n' > "$TLEDGER"
printf '0\tother-island\texpX\tm\t0\t0\t0\t0\t100.0\n' >> "$TLEDGER"
OUT=$(env $LOOP_ENV MOCK_PLAN="improve" ANCHOR=0 MAX_EXPS=5 STAG_N=99 COST_CAP=40 bash "$SCRIPT_DIR/loop.sh" "$ISL" 2>&1)
echo "$OUT" | grep -q "STOP: COST_CAP" && ok "cost cap stop (shared ledger)" || bad "no COST_CAP stop"
[ "$(rows)" -eq 0 ] && ok "zero sessions spawned past cap" || bad "rows=$(rows) after cap (expected 0)"

# ---------------------------------------------------------------- L4
hdr "L4: failed submission -> next session gets diagnose framing"
reset_island
mkdir -p "$ISL/workspace/agent0"
echo "engine-exit-1 (fake for test)" > "$ISL/workspace/agent0/RESULT_ERROR.999"
OUT=$(env $LOOP_ENV MOCK_PLAN="nosubmit improve" ANCHOR=0 MAX_EXPS=1 STAG_N=99 bash "$SCRIPT_DIR/loop.sh" "$ISL" 2>&1)
grep -q "Oracle error" "$ISL/.agent_prompts/session_exp1.md" 2>/dev/null \
    && ok "second build of exp1 prompt carries oracle-error framing" || bad "no error framing in rebuilt prompt"
[ "$(rows)" -eq 1 ] && ok "recovered and logged the experiment" || bad "rows=$(rows) (expected 1)"

# ---------------------------------------------------------------- L5
hdr "L5: idempotence — L1 sequence twice from clean resets"
reset_island
env $LOOP_ENV MOCK_PLAN="improve" ANCHOR=1 MAX_EXPS=3 STAG_N=99 bash "$SCRIPT_DIR/loop.sh" "$ISL" > /dev/null 2>&1
R1=$(rows)
reset_island
env $LOOP_ENV MOCK_PLAN="improve" ANCHOR=1 MAX_EXPS=3 STAG_N=99 bash "$SCRIPT_DIR/loop.sh" "$ISL" > /dev/null 2>&1
R2=$(rows)
[ "$R1" -eq 3 ] && [ "$R2" -eq 3 ] && ok "identical results across clean resets ($R1/$R2)" || bad "runs differ: $R1 vs $R2"

# ---------------------------------------------------------------- summary
echo
echo "================================================================"
echo "loop-preflight: $PASS ok, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    echo "RESULT: FAIL"
    exit 1
fi
rm -f "$TLEDGER"
echo "RESULT: PASS — v5.1 lifecycle verified with \$0 of model spend."
