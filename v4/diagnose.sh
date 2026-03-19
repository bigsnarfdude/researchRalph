#!/bin/bash
# diagnose.sh — compute process quality and stopping decision
#
# Reads blackboard.md, results.tsv, meta-blackboard.md, sae.py
# Outputs a JSON-ish report to stderr and a single decision to stdout:
#   CONTINUE | STOP_HACKING | STOP_DONE | REDESIGN | TOO_EARLY
#
# Usage: bash diagnose.sh /path/to/domain

DOMAIN_DIR="${1:-.}"

# --- Count experiments ---
TOTAL_EXP=$(grep -c $'\t' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)

if [ "$TOTAL_EXP" -lt 8 ]; then
    echo "[diagnose] Too early ($TOTAL_EXP experiments)" >&2
    echo "TOO_EARLY"
    exit 0
fi

# --- Process quality signals ---
BB="$DOMAIN_DIR/blackboard.md"
SAE="$DOMAIN_DIR/sae.py"
META="$DOMAIN_DIR/meta-blackboard.md"
RESULTS="$DOMAIN_DIR/results.tsv"

# Papers/references cited
PAPERS=$(grep -ciE "arxiv|paper|et al\.|journal|conference|ICML|NeurIPS|ICLR|gregor|lecun" "$BB" 2>/dev/null || echo 0)

# Architecture classes defined
CLASSES=$(grep -c "^class " "$SAE" 2>/dev/null || echo 0)

# Explanatory reasoning in blackboard
EXPLANATIONS=$(grep -ciE "because|mechanism|why this|the reason|this works because|hypothesis" "$BB" 2>/dev/null || echo 0)

# Blackboard total lines (proxy for effort)
BB_LINES=$(wc -l < "$BB" 2>/dev/null | tr -d ' ')

# Distinct experiments (non-duplicate descriptions)
UNIQUE_DESIGNS=$(awk -F'\t' '{print $7}' "$RESULTS" 2>/dev/null | sort -u | wc -l | tr -d ' ')

# Ablation signals (experiments explicitly called ablation or "same as X but")
ABLATIONS=$(grep -ciE "ablation|same as .* but|removed|without|vs\.|compared to" "$BB" 2>/dev/null || echo 0)

# Simplification signals
SIMPLIFICATIONS=$(grep -ciE "simpler|removed|dropped|back to|fewer|1-step|simplified" "$BB" 2>/dev/null || echo 0)

# --- Score process quality (0-30) ---
PQ=0
[ "$PAPERS" -gt 0 ]          && PQ=$((PQ + 3))
[ "$PAPERS" -gt 3 ]          && PQ=$((PQ + 3))
[ "$CLASSES" -gt 1 ]          && PQ=$((PQ + 3))
[ "$CLASSES" -gt 5 ]          && PQ=$((PQ + 3))
[ "$EXPLANATIONS" -gt 3 ]     && PQ=$((PQ + 3))
[ "$EXPLANATIONS" -gt 10 ]    && PQ=$((PQ + 3))
[ "$ABLATIONS" -gt 0 ]        && PQ=$((PQ + 3))
[ "$ABLATIONS" -gt 3 ]        && PQ=$((PQ + 3))
[ "$SIMPLIFICATIONS" -gt 0 ]  && PQ=$((PQ + 3))
[ "$UNIQUE_DESIGNS" -gt 5 ]   && PQ=$((PQ + 3))

# --- Score trajectory ---
# Best score overall
BEST_SCORE=$(awk -F'\t' '{print $2}' "$RESULTS" 2>/dev/null | sort -rn | head -1)

# Best score in last 10 experiments
RECENT_BEST=$(tail -10 "$RESULTS" | awk -F'\t' '{print $2}' | sort -rn | head -1)

# Best score in experiments before the last 10
PRIOR_BEST=$(head -n -10 "$RESULTS" 2>/dev/null | awk -F'\t' '{print $2}' | sort -rn | head -1)

# Check if flat (< 0.5% improvement in last 10 experiments)
if [ -n "$PRIOR_BEST" ] && [ -n "$RECENT_BEST" ]; then
    # Use awk for float comparison
    FLAT=$(awk "BEGIN {
        delta = $RECENT_BEST - $PRIOR_BEST;
        if (delta < 0.005) print \"true\"; else print \"false\"
    }")
else
    FLAT="false"
fi

# --- Blind spots ---
BLIND_SPOTS=0
if [ -f "$META" ]; then
    # Count items in blind spots section
    BLIND_SPOTS=$(awk '/^## [Bb]lind [Ss]pots/,/^## /' "$META" 2>/dev/null | grep -c "^- " || echo 0)
fi

# --- Stagnation depth ---
# How many consecutive experiments since last best?
STAGNATION=0
if [ "$TOTAL_EXP" -gt 0 ]; then
    STAGNATION=$(awk -F'\t' -v best="$BEST_SCORE" '
        BEGIN { since=0 }
        { since++; if ($2 == best) since=0 }
        END { print since }
    ' "$RESULTS")
fi

# --- Report ---
echo "[diagnose] === Process Quality Report ===" >&2
echo "[diagnose] Experiments: $TOTAL_EXP (unique designs: $UNIQUE_DESIGNS)" >&2
echo "[diagnose] Papers cited: $PAPERS" >&2
echo "[diagnose] Architecture classes: $CLASSES" >&2
echo "[diagnose] Explanations: $EXPLANATIONS" >&2
echo "[diagnose] Ablations: $ABLATIONS" >&2
echo "[diagnose] Simplifications: $SIMPLIFICATIONS" >&2
echo "[diagnose] Process quality: $PQ / 30" >&2
echo "[diagnose] Best score: $BEST_SCORE" >&2
echo "[diagnose] Score flat (last 10): $FLAT" >&2
echo "[diagnose] Stagnation depth: $STAGNATION experiments since best" >&2
echo "[diagnose] Blind spots: $BLIND_SPOTS" >&2
echo "[diagnose] ===" >&2

# --- Decision matrix ---
if [ "$PQ" -lt 10 ] && [ "$TOTAL_EXP" -gt 15 ]; then
    echo "[diagnose] DECISION: STOP_HACKING (low process quality after $TOTAL_EXP experiments)" >&2
    echo "STOP_HACKING"
elif [ "$FLAT" = "true" ] && [ "$PQ" -ge 10 ] && [ "$BLIND_SPOTS" -eq 0 ] && [ "$STAGNATION" -gt 15 ]; then
    echo "[diagnose] DECISION: STOP_DONE (high quality, flat, no blind spots, stagnation=$STAGNATION)" >&2
    echo "STOP_DONE"
elif [ "$FLAT" = "true" ] && [ "$PQ" -ge 10 ] && [ "$BLIND_SPOTS" -gt 0 ] && [ "$STAGNATION" -gt 15 ]; then
    echo "[diagnose] DECISION: REDESIGN (high quality, flat, $BLIND_SPOTS blind spots remain)" >&2
    echo "REDESIGN"
else
    echo "[diagnose] DECISION: CONTINUE" >&2
    echo "CONTINUE"
fi
