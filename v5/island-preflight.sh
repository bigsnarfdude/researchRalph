#!/bin/bash
# island-preflight.sh — v5.0 mechanics test suite (T0–T5). $0 of agent spend.
#
# Proves the island plumbing with stub workers and a canned advisor before any
# real launch. Covers the erdos-125 failure class (mechanical wiring) plus the
# three new island surfaces: board partitioning, per-island diagnosis, migration.
#
# Usage: bash v5/island-preflight.sh [/path/to/base-domain]
#   Base domain defaults to domains/island-mock. Islands are recreated from
#   scratch every run (clean fixture reset), so the suite is idempotent.
# Exit 0 = all tests pass. Exit 1 = at least one failure — do not launch.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BASE="${1:-$REPO_ROOT/domains/island-mock}"
BASE="$(cd "$BASE" && pwd)"
BASE_NAME="$(basename "$BASE")"
PARENT="$(dirname "$BASE")"
ISL_A="$PARENT/$BASE_NAME-isl-a"; ISL_B="$PARENT/$BASE_NAME-isl-b"; ISL_C="$PARENT/$BASE_NAME-isl-c"

PASS=0; FAIL=0; declare -a FAILURES=()
ok()   { echo "  ok: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); FAILURES+=("$1"); }
hdr()  { echo; echo "== $1"; }

data_rows() { awk 'NR>1' "$1/results.tsv" 2>/dev/null | wc -l | tr -d ' '; }

strip_last_row() {  # portable (GNU/BSD) + preserves file mode (444-protected domains)
    python3 - "$1" <<'PY'
import os, stat, sys
p = sys.argv[1]
mode = stat.S_IMODE(os.stat(p).st_mode)
os.chmod(p, 0o644)
with open(p) as f:
    lines = f.readlines()
with open(p, "w") as f:
    f.writelines(lines[:-1])
os.chmod(p, mode)
PY
}

# ---------------------------------------------------------------- setup
hdr "setup: clean island reset (3 islands from $BASE_NAME)"
bash "$SCRIPT_DIR/make-islands.sh" "$BASE" 3 --force || { echo "FATAL: make-islands failed"; exit 1; }

# ---------------------------------------------------------------- T0
hdr "T0: existing preflight holds, per island (static + live oracle)"
for ISL in "$ISL_A" "$ISL_B" "$ISL_C"; do
    N="$(basename "$ISL")"
    if bash "$REPO_ROOT/v4/preflight.sh" "$ISL" > /dev/null 2>&1; then
        ok "v4/preflight.sh passes on $N"
    else
        bad "v4/preflight.sh fails on $N"
    fi
    # Live oracle test (fixture is not lean_proof, so preflight skips it — run it here)
    mkdir -p "$ISL/workspace/agent0"
    EDITABLE="$(grep '^editable:' "$ISL/config.yaml" 2>/dev/null | awk '{print $2}')"
    EDITABLE="${EDITABLE:-answer.txt}"
    if [ -f "$ISL/$EDITABLE" ]; then
        cp "$ISL/$EDITABLE" "$ISL/workspace/agent0/$EDITABLE"
    else
        printf 'preflight probe answer with a handful of words\n' > "$ISL/workspace/agent0/$EDITABLE"
    fi
    BEFORE=$(data_rows "$ISL")
    OUT=$(cd "$ISL" && CLAUDE_AGENT_ID=agent0 bash run.sh probe "preflight probe" 2>&1)
    if echo "$OUT" | grep -q "workspace/agent0"; then
        ok "oracle on $N reads workspace/agent0 (SOURCE line present)"
    else
        bad "oracle on $N never mentions workspace/agent0 — reading the wrong file"
    fi
    AFTER=$(data_rows "$ISL")
    if [ "$AFTER" -eq $((BEFORE+1)) ]; then
        ok "oracle on $N logged a row to results.tsv"
        strip_last_row "$ISL/results.tsv"   # remove probe row, same as v4/preflight.sh
    else
        bad "oracle on $N logged no row (before=$BEFORE after=$AFTER)"
    fi
done

# ---------------------------------------------------------------- stub launch
hdr "launch: 3 islands x 2 stub workers (6 board writes, 6 oracle calls)"
for ISL in "$ISL_A" "$ISL_B" "$ISL_C"; do
    S="${ISL##*-isl-}"
    for AG in agent0 agent1; do
        if ! ISLAND="$S" AGENT="$AG" DOMAIN="$ISL" bash "$SCRIPT_DIR/mock-worker.sh" > /dev/null 2>&1; then
            bad "stub $S/$AG failed"
        fi
    done
done
echo "  (stubs done)"

# ---------------------------------------------------------------- T1
hdr "T1: board isolation — no cross-island writes"
CROSS=0
for S in a b c; do
    for OTHER in "$ISL_A" "$ISL_B" "$ISL_C"; do
        [ "${OTHER##*-isl-}" = "$S" ] && continue
        if grep -rq "mock-${S}-" "$OTHER" 2>/dev/null; then
            bad "marker mock-${S}-* leaked into $(basename "$OTHER")"; CROSS=1
        fi
    done
done
[ "$CROSS" -eq 0 ] && ok "zero cross-island marker leaks"
for ISL in "$ISL_A" "$ISL_B" "$ISL_C"; do
    S="${ISL##*-isl-}"
    CNT=$(grep -c "mock-${S}-" "$ISL/blackboard.md" 2>/dev/null)
    if [ "$CNT" -eq 2 ]; then
        ok "island $S board has exactly its own 2 markers"
    else
        bad "island $S board has $CNT own markers (expected 2)"
    fi
done

# ---------------------------------------------------------------- T2
hdr "T2: results are per-island and aggregate to 2/2/2"
AGG_OK=1
for ISL in "$ISL_A" "$ISL_B" "$ISL_C"; do
    S="${ISL##*-isl-}"
    ROWS=$(data_rows "$ISL")
    if [ "$ROWS" -eq 2 ]; then
        ok "island $S results.tsv has 2 data rows"
    else
        bad "island $S results.tsv has $ROWS rows (expected 2)"; AGG_OK=0
    fi
    AGENTS=$(awk -F'\t' 'NR>1 {print $6}' "$ISL/results.tsv" | sort | paste -sd, -)
    if [ "$AGENTS" = "agent0,agent1" ]; then
        ok "island $S rows tagged agent0 + agent1"
    else
        bad "island $S agent column is '$AGENTS' (expected agent0,agent1)"; AGG_OK=0
    fi
done
if [ "$AGG_OK" -eq 1 ]; then
    # island identity derives from the directory — the aggregation view the gardener uses
    for ISL in "$ISL_A" "$ISL_B" "$ISL_C"; do
        awk -F'\t' -v i="${ISL##*-isl-}" 'NR>1 {print i"\t"$0}' "$ISL/results.tsv"
    done | awk -F'\t' '{n[$1]++} END {printf "  aggregate by island:"; for (k in n) printf " %s=%s", k, n[k]; print ""}'
fi

# ---------------------------------------------------------------- T3
hdr "T3: diagnose.py on island a provably never reads island b"
TMP="${TMPDIR:-/tmp}"
D_BASE=$(python3 "$REPO_ROOT/v4/diagnose.py" "$ISL_A" 2>"$TMP/diag_base.txt"); RC1=$?
if [ $RC1 -ne 0 ] || [ -z "$D_BASE" ]; then
    bad "diagnose.py failed on island a (rc=$RC1) — cannot verify isolation"
else
    ok "diagnose.py runs on island a (decision: $D_BASE)"
    mv "$ISL_B/blackboard.md" "$ISL_B/blackboard.md.hidden"
    D_HID=$(python3 "$REPO_ROOT/v4/diagnose.py" "$ISL_A" 2>"$TMP/diag_hidden.txt"); RC2=$?
    mv "$ISL_B/blackboard.md.hidden" "$ISL_B/blackboard.md"
    if [ $RC2 -eq 0 ] && [ "$D_BASE" = "$D_HID" ] && diff -q "$TMP/diag_base.txt" "$TMP/diag_hidden.txt" > /dev/null; then
        ok "identical decision AND full report with island b's board hidden ($D_HID)"
    else
        bad "diagnose changed with b hidden (rc=$RC2, '$D_BASE' vs '$D_HID') — cross-island read"
    fi
fi

# ---------------------------------------------------------------- T4
hdr "T4: line budget fires (warn + flag, never reject)"
{ printf '# Blackboard — seeded over budget\n'; printf -- '- filler line\n%.0s' $(seq 1 349); } > "$ISL_A/blackboard.md"
if ISLAND=a AGENT=agent0 DOMAIN="$ISL_A" bash "$SCRIPT_DIR/mock-worker.sh" > /dev/null 2>&1; then
    ok "stub write still succeeds over budget (non-blocking by design)"
else
    bad "stub write failed on over-budget board — enforcement is blocking, not warn+flag"
fi
bash "$SCRIPT_DIR/board-budget.sh" "$ISL_A" > /dev/null 2>&1; RC=$?
if [ $RC -eq 2 ] && [ -f "$ISL_A/BOARD_OVER_BUDGET" ]; then
    ok "over-budget board: exit 2 + BOARD_OVER_BUDGET flag set ($(wc -l < "$ISL_A/blackboard.md" | tr -d ' ') lines)"
else
    bad "over-budget board: rc=$RC, flag $([ -f "$ISL_A/BOARD_OVER_BUDGET" ] && echo present || echo missing)"
fi
bash "$SCRIPT_DIR/board-budget.sh" "$ISL_B" > /dev/null 2>&1; RC=$?
if [ $RC -eq 0 ] && [ ! -f "$ISL_B/BOARD_OVER_BUDGET" ]; then
    ok "under-budget board: exit 0, no flag"
else
    bad "under-budget board: rc=$RC (expected 0)"
fi

# ---------------------------------------------------------------- T5
hdr "T5: migration end-to-end with canned advisor (ADVISOR_STUB)"
ADVISOR_STUB="$SCRIPT_DIR/fixtures/canned_digest.md" \
    bash "$SCRIPT_DIR/migrate.sh" --from "$ISL_A" --to "$ISL_B" "$ISL_C" > /dev/null 2>&1
M="<!-- DIGEST(from=$BASE_NAME-isl-a) -->"
CB=$(grep -cF "$M" "$ISL_B/blackboard.md"); CC=$(grep -cF "$M" "$ISL_C/blackboard.md"); CA=$(grep -cF "$M" "$ISL_A/blackboard.md")
if [ "$CB" -eq 1 ] && [ "$CC" -eq 1 ]; then
    ok "digest from a landed exactly once on b and c"
else
    bad "digest counts wrong (b=$CB c=$CC, expected 1/1)"
fi
if [ "$CA" -eq 0 ]; then
    ok "no self-migration onto a"
else
    bad "digest self-injected onto a ($CA)"
fi
ADVISOR_STUB="$SCRIPT_DIR/fixtures/canned_digest.md" \
    bash "$SCRIPT_DIR/migrate.sh" --from "$ISL_A" --to "$ISL_B" "$ISL_C" > /dev/null 2>&1
CB2=$(grep -cF "$M" "$ISL_B/blackboard.md"); CC2=$(grep -cF "$M" "$ISL_C/blackboard.md")
if [ "$CB2" -eq 1 ] && [ "$CC2" -eq 1 ]; then
    ok "re-run is idempotent (still exactly one digest each)"
else
    bad "re-run duplicated digests (b=$CB2 c=$CC2)"
fi
if [ ! -f "$ISL_B/BOARD_OVER_BUDGET" ] && [ ! -f "$ISL_C/BOARD_OVER_BUDGET" ]; then
    ok "budgets charged post-migration: b and c still under cap"
else
    bad "migration blew a budget flag on b or c"
fi

# ---------------------------------------------------------------- summary
echo
echo "================================================================"
echo "island-preflight: $PASS ok, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    echo "RESULT: FAIL — do not launch islands until fixed."
    exit 1
fi
echo "RESULT: PASS — island mechanics verified with \$0 of agent spend."
