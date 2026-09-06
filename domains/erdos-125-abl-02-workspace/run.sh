#!/bin/bash
# Oracle for Erdős #125 — binary: proof compiles without sorry or it doesn't
# Repaired 2026-09-06: d923460 spliced the RH-prevention block INSIDE the `if`
# branch and truncated the logging tail, leaving the file unparseable. The block
# now sits after the `fi`, error paths are explicit, and the Lean project path is
# taken from the environment instead of a hardcoded host path.
set -e

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LEAN_PROJECT="${RRMA_LEAN_PROJECT:-$HOME/rrma-lean}"
AGENT="${CLAUDE_AGENT_ID:-agent0}"
export PATH="$HOME/.elan/bin:$PATH"

# ABLATION 02: workspace resolution removed — always reads the domain root,
# ignores workspace/ entirely.
SOLUTION="$DOMAIN_DIR/Erdos125.lean"

if [ ! -f "$SOLUTION" ]; then
    echo "SCORE=0.0"
    echo "STATUS: Erdos125.lean not found"
    exit 0
fi

# Count sorries (exclude comment lines). grep -c exits 1 on zero matches.
SORRY_COUNT=$(grep -v '^\s*--' "$SOLUTION" | grep -c "sorry" 2>/dev/null) || true
SORRY_COUNT=${SORRY_COUNT:-0}

TMP_FILE="$LEAN_PROJECT/Erdos125Test_${AGENT}_$$.lean"
cp "$SOLUTION" "$TMP_FILE"
cd "$LEAN_PROJECT"
set +e
BUILD_OUT=$(lake env lean "$TMP_FILE" 2>&1)
BUILD_EXIT=$?
set -e
rm -f "$TMP_FILE"

echo "=== ORACLE ==="
echo "SORRY_COUNT: $SORRY_COUNT"
echo "BUILD_EXIT: $BUILD_EXIT"
echo "SOURCE: $SOLUTION"

# Optional integrity gate — OFF by default so it cannot alter ablation semantics.
# RRMA_AXIOM_GATE=1 rejects a win whose proof leans on anything beyond the
# standard three axioms (catches `axiom` decls and native_decide's ofReduceBool).
AXIOM_OK=1
if [ "${RRMA_AXIOM_GATE:-0}" = "1" ] && [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    GATE_FILE="$LEAN_PROJECT/AxiomGate_${AGENT}_$$.lean"
    { cat "$SOLUTION"; echo; echo "#print axioms erdos_125"; } > "$GATE_FILE"
    set +e
    GATE_OUT=$(lake env lean "$GATE_FILE" 2>&1)
    set -e
    rm -f "$GATE_FILE"
    # Whitelist, not blacklist. A blacklist missed `exists_k_m_ratio_close`, the one
    # real user axiom in the corpus (erdos-125-abl-07). Anything outside Lean's three
    # standard axioms fails; native_decide's Lean.ofReduceBool fails with it.
    if echo "$GATE_OUT" | grep -q "does not depend on any axioms"; then
        echo "AXIOM_GATE: pass (no axioms)"
    elif ! echo "$GATE_OUT" | grep -q "depends on axioms"; then
        AXIOM_OK=0
        echo "AXIOM_GATE: FAIL — no axiom report; gate file did not elaborate"
    else
        AXS=$(echo "$GATE_OUT" | sed -n 's/.*depends on axioms: \[\(.*\)\].*/\1/p' | tr -d ' ' | tr ',' '\n')
        BAD=$(echo "$AXS" | grep -v '^$' | grep -vxE 'propext|Classical\.choice|Quot\.sound' || true)
        if [ -n "$BAD" ]; then
            AXIOM_OK=0
            echo "AXIOM_GATE: FAIL — non-standard axioms: $(echo "$BAD" | tr '\n' ' ')"
        else
            echo "AXIOM_GATE: pass"
        fi
    fi
fi

MAX_SORRY=4
if [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ] && [ "$AXIOM_OK" -eq 1 ]; then
    echo "SCORE=1.0"; echo "STATUS: PROVED"
    SCORE="1.0"; STATUS="proved"
elif [ "$SORRY_COUNT" -eq 0 ] && [ "$BUILD_EXIT" -eq 0 ]; then
    echo "SCORE=0.0"; echo "STATUS: AXIOM_GATE_REJECTED"
    SCORE="0.0"; STATUS="axiom_rejected"
elif [ "$BUILD_EXIT" -ne 0 ]; then
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: COMPILE_ERROR — $SORRY_COUNT sorry, build failed"
    echo "$BUILD_OUT" | tail -30
    SCORE="$FRAC"; STATUS="compile_error"
else
    FRAC=$(echo "scale=3; ($MAX_SORRY - $SORRY_COUNT) / $MAX_SORRY" | bc 2>/dev/null || echo "0.0")
    echo "SCORE=$FRAC"; echo "STATUS: IN_PROGRESS — $SORRY_COUNT sorry remaining"
    echo "$BUILD_OUT" | grep "error:\|warning:\|sorry" | head -20
    SCORE="$FRAC"; STATUS="in_progress"
fi

# --- RH Prevention: flock + chmod ensures only the oracle writes results.tsv ---
# Description is oracle-generated only; agent-supplied text is never logged, so
# every row has unambiguous provenance.
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DESCRIPTION="proof attempt $TIMESTAMP"
RESULTS="$DOMAIN_DIR/results.tsv"
LOCK="$DOMAIN_DIR/results.lock"
(
    flock -x -w 30 200 || { echo "[oracle] could not acquire results.lock — not logged"; exit 0; }
    chmod 644 "$RESULTS" 2>/dev/null || true
    if [ ! -f "$RESULTS" ] || ! head -1 "$RESULTS" | grep -q "^EXP-ID"; then
        printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "$RESULTS"
    fi
    EXP_ID="exp$(printf '%03d' $(( $(wc -l < "$RESULTS") )))"
    printf "%s\t%s\t%s\t%s\t%s\n" "$EXP_ID" "$SCORE" "$STATUS" "$DESCRIPTION" "$AGENT" >> "$RESULTS"
    chmod 444 "$RESULTS"
) 200>"$LOCK"
