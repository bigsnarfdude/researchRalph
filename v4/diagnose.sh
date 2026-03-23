#!/bin/bash
# diagnose.sh — compute process quality and stopping decision (v4.2)
#
# Reads blackboard.md, results.tsv, meta-blackboard.md
# Outputs a report to stderr and a single decision to stdout:
#   CONTINUE | NUDGE | STOP_HACKING | STOP_DONE | REDESIGN | TOO_EARLY
#
# v4.1 changes:
#   - NUDGE state (between CONTINUE and REDESIGN)
#   - Rolling window stagnation (1% improvement over last 20 experiments)
#   - Axis diversity in recent experiments
#   - Domain-agnostic PQ (no sae.py dependency)
#
# v4.2 changes:
#   - NUDGE now fires BEFORE STOP_DONE/REDESIGN (was unreachable in v4.1)
#   - NUDGE condition 4: explicit FLAT=false guard (stagnation without rolling-flat)
#   - AXIS_DIVERSE threshold relaxed: ≤3 designs (was ≤2)
#
# Usage: bash diagnose.sh /path/to/domain

DOMAIN_DIR="${1:-.}"

# --- Count experiments ---
TOTAL_EXP=$(tail -n +2 "$DOMAIN_DIR/results.tsv" 2>/dev/null | awk 'NF{c++} END{print c+0}')

if [ "$TOTAL_EXP" -lt 8 ]; then
    echo "[diagnose] Too early ($TOTAL_EXP experiments)" >&2
    echo "TOO_EARLY"
    exit 0
fi

# --- Artifact paths ---
BB="$DOMAIN_DIR/blackboard.md"
META="$DOMAIN_DIR/meta-blackboard.md"
RESULTS="$DOMAIN_DIR/results.tsv"

# --- Process quality signals (domain-agnostic) ---
# NOTE: all counts use awk to avoid grep -c multiline/exit-code issues on Ubuntu

# Papers/references cited in blackboard
PAPERS=$(awk 'BEGIN{IGNORECASE=1} /arxiv|paper|et al\.|journal|conference|ICML|NeurIPS|ICLR/{c++} END{print c+0}' "$BB" 2>/dev/null)

# Explanatory reasoning (why/because/mechanism/analysis/insight/finding)
EXPLANATIONS=$(awk 'BEGIN{IGNORECASE=1} /because|mechanism|why |the reason|hypothesis|theory|key insight|key finding|observation|confirms|suggests|indicates|implies|conclusion/{c++} END{print c+0}' "$BB" 2>/dev/null)

# Ablation/comparison signals (includes data-driven comparisons)
ABLATIONS=$(awk 'BEGIN{IGNORECASE=1} /ablation|same as .* but|without|vs |compared to|isolat|worse than|better than|no improvement|WORSE|BETTER|sweep|>> |<< /{c++} END{print c+0}' "$BB" 2>/dev/null)

# Simplification signals
SIMPLIFICATIONS=$(awk 'BEGIN{IGNORECASE=1} /simpler|removed|dropped|back to|fewer|simplified|unnecessary|disabl|turning off|not needed/{c++} END{print c+0}' "$BB" 2>/dev/null)

# Unique experiment designs
UNIQUE_DESIGNS=$(tail -n +2 "$RESULTS" 2>/dev/null | awk -F'\t' '{print $7}' | sort -u | awk 'NF{c++} END{print c+0}')

# Architecture classes (if sae.py exists — bonus, not required)
CLASSES=0
for f in "$DOMAIN_DIR/sae.py" "$DOMAIN_DIR/"*.py; do
    if [ -f "$f" ] && [ "$(basename "$f")" != "engine.py" ]; then
        C=$(grep -c "^class " "$f" 2>/dev/null | tr -d '\n' || echo 0)
        CLASSES=$((CLASSES + C))
    fi
done

# Blackboard depth (reasoning density = explanations / total lines)
BB_LINES=$(wc -l < "$BB" 2>/dev/null | tr -d ' ' || echo 1)

# --- Axis diversity in last 10 experiments ---
# Count unique design categories in recent experiments
RECENT_DESIGNS=$(tail -10 "$RESULTS" | awk -F'\t' '{print $7}' | sort -u | awk 'NF{c++} END{print c+0}')
AXIS_DIVERSE="true"
if [ "$RECENT_DESIGNS" -le 3 ] && [ "$TOTAL_EXP" -gt 15 ]; then
    AXIS_DIVERSE="false"
fi

# --- Score process quality (0-30) ---
PQ=0
[ "${PAPERS:-0}" -gt 0 ]          && PQ=$((PQ + 3))
[ "${PAPERS:-0}" -gt 3 ]          && PQ=$((PQ + 3))
[ "${EXPLANATIONS:-0}" -gt 3 ]    && PQ=$((PQ + 3))
[ "${EXPLANATIONS:-0}" -gt 10 ]   && PQ=$((PQ + 3))
[ "${ABLATIONS:-0}" -gt 0 ]       && PQ=$((PQ + 3))
[ "${ABLATIONS:-0}" -gt 3 ]       && PQ=$((PQ + 3))
[ "${SIMPLIFICATIONS:-0}" -gt 0 ] && PQ=$((PQ + 3))
[ "${UNIQUE_DESIGNS:-0}" -gt 5 ]  && PQ=$((PQ + 3))
[ "${CLASSES:-0}" -gt 1 ]         && PQ=$((PQ + 3))
[ "${BB_LINES:-0}" -gt 100 ]      && PQ=$((PQ + 3))

# --- Score trajectory ---
BEST_SCORE=$(tail -n +2 "$RESULTS" 2>/dev/null | awk -F'\t' '{print $2}' | sort -rn | head -1)

# Rolling window: best in last 20 vs best in prior experiments
BEST_LAST_20=$(tail -20 "$RESULTS" | awk -F'\t' '{print $2}' | sort -rn | head -1)
BEST_PRIOR_20=$(head -n -20 "$RESULTS" 2>/dev/null | tail -n +2 | awk -F'\t' '{print $2}' | sort -rn | head -1)

# Flat = less than 1% absolute improvement over rolling window of 20
FLAT="false"
if [ -n "$BEST_PRIOR_20" ] && [ -n "$BEST_LAST_20" ] && [ "$TOTAL_EXP" -gt 20 ]; then
    FLAT=$(awk "BEGIN {
        delta = $BEST_LAST_20 - $BEST_PRIOR_20;
        if (delta < 0.01) print \"true\"; else print \"false\"
    }")
fi

# Also check shorter window: best in last 10 vs best overall
BEST_LAST_10=$(tail -10 "$RESULTS" | awk -F'\t' '{print $2}' | sort -rn | head -1)
MICRO_FLAT="false"
if [ -n "$BEST_SCORE" ] && [ -n "$BEST_LAST_10" ] && [ "$TOTAL_EXP" -gt 15 ]; then
    MICRO_FLAT=$(awk "BEGIN {
        delta = $BEST_LAST_10 - ($BEST_SCORE - 0.005);
        if (delta < 0.005) print \"true\"; else print \"false\"
    }")
fi

# --- Stagnation depth (experiments since best) ---
STAGNATION=0
if [ "$TOTAL_EXP" -gt 0 ]; then
    STAGNATION=$(tail -n +2 "$RESULTS" | awk -F'\t' -v best="$BEST_SCORE" '
        BEGIN { since=0 }
        { since++; if ($2 == best) since=0 }
        END { print since }
    ')
fi

# --- Blind spots ---
BLIND_SPOTS=0
if [ -f "$META" ]; then
    BLIND_SPOTS=$(awk '/^## [Bb]lind [Ss]pots/,/^## /' "$META" 2>/dev/null | awk '/^- /{c++} END{print c+0}')
fi

# --- Report ---
echo "[diagnose] === Process Quality Report (v4.1) ===" >&2
echo "[diagnose] Experiments: $TOTAL_EXP (unique designs: $UNIQUE_DESIGNS)" >&2
echo "[diagnose] Papers cited: $PAPERS" >&2
echo "[diagnose] Code classes: $CLASSES" >&2
echo "[diagnose] Explanations: $EXPLANATIONS" >&2
echo "[diagnose] Ablations: $ABLATIONS" >&2
echo "[diagnose] Simplifications: $SIMPLIFICATIONS" >&2
echo "[diagnose] Blackboard lines: $BB_LINES" >&2
echo "[diagnose] Process quality: $PQ / 30" >&2
echo "[diagnose] Best score: $BEST_SCORE" >&2
echo "[diagnose] Flat (1% over last 20): $FLAT" >&2
echo "[diagnose] Micro-flat (last 10 near best): $MICRO_FLAT" >&2
echo "[diagnose] Stagnation depth: $STAGNATION exp since best" >&2
echo "[diagnose] Recent axis diversity: $RECENT_DESIGNS designs in last 10 ($AXIS_DIVERSE)" >&2
echo "[diagnose] Blind spots: $BLIND_SPOTS" >&2
echo "[diagnose] ===" >&2

# --- Decision matrix (v4.2) ---
# NUDGE fires before STOP_DONE/REDESIGN so local stagnation gets an intervention
# before the full rolling-window flat threshold is reached.

# 1. Too few experiments
if [ "$TOTAL_EXP" -lt 8 ]; then
    echo "[diagnose] DECISION: TOO_EARLY" >&2
    echo "TOO_EARLY"

# 2. Low process quality = hacking
elif [ "$PQ" -lt 10 ] && [ "$TOTAL_EXP" -gt 15 ]; then
    echo "[diagnose] DECISION: STOP_HACKING (PQ=$PQ after $TOTAL_EXP experiments)" >&2
    echo "STOP_HACKING"

# 3. High PQ + micro-flat + low axis diversity = NUDGE (early intervention)
elif [ "$MICRO_FLAT" = "true" ] && [ "$AXIS_DIVERSE" = "false" ] && [ "$PQ" -ge 10 ]; then
    echo "[diagnose] DECISION: NUDGE (PQ=$PQ, micro-flat, low axis diversity: $RECENT_DESIGNS designs in last 10)" >&2
    echo "NUDGE"

# 4. High PQ + stagnation > 10 but not yet rolling-flat = NUDGE (catch stalls before STOP)
elif [ "$STAGNATION" -gt 10 ] && [ "$FLAT" = "false" ] && [ "$PQ" -ge 10 ]; then
    echo "[diagnose] DECISION: NUDGE (PQ=$PQ, stagnation=$STAGNATION, not yet flat)" >&2
    echo "NUDGE"

# 5. High PQ + flat + no blind spots + deep stagnation = done
elif [ "$FLAT" = "true" ] && [ "$PQ" -ge 10 ] && [ "$BLIND_SPOTS" -eq 0 ] && [ "$STAGNATION" -gt 10 ]; then
    echo "[diagnose] DECISION: STOP_DONE (PQ=$PQ, flat, no blind spots, stagnation=$STAGNATION)" >&2
    echo "STOP_DONE"

# 6. High PQ + flat + blind spots = redesign scaffold
elif [ "$FLAT" = "true" ] && [ "$PQ" -ge 10 ] && [ "$BLIND_SPOTS" -gt 0 ] && [ "$STAGNATION" -gt 10 ]; then
    echo "[diagnose] DECISION: REDESIGN (PQ=$PQ, flat, $BLIND_SPOTS blind spots, stagnation=$STAGNATION)" >&2
    echo "REDESIGN"

# 7. Default
else
    echo "[diagnose] DECISION: CONTINUE" >&2
    echo "CONTINUE"
fi
