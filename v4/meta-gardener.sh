#!/bin/bash
# meta-gardener.sh — seeds a new domain from prior run artifacts
#
# The inception loop:
#   prior run artifacts → meta-gardener → optimized program.md → outer-loop → agents → results
#                              ↑                                                           |
#                              └──────────────────────────────────────────────────────────┘
#
# The meta-gardener is NOT the gardener. It doesn't run experiments or monitor.
# It does one thing: given what was learned, write the best possible starting
# conditions for the next run. Then it hands off to outer-loop.sh.
#
# Usage:
#   bash meta-gardener.sh <prior_domain> <new_domain> [max_gens] [num_agents] [max_turns] [monitor_min]
#
# The new domain must already exist with: config.yaml, engine.py, run.sh, sae.py,
# blackboard.md — everything except a seeded program.md (meta-gardener writes that).
#
# If new_domain already has a program.md, meta-gardener rewrites it.

set -euo pipefail

PRIOR_DIR="${1:-}"
NEW_DIR="${2:-}"

if [ -z "$PRIOR_DIR" ] || [ -z "$NEW_DIR" ]; then
    echo "Usage: bash meta-gardener.sh <prior_domain> <new_domain> [max_gens] [num_agents] [max_turns] [monitor_min]"
    exit 1
fi

PRIOR_DIR="$(cd "$PRIOR_DIR" && pwd)"
NEW_DIR="$(cd "$NEW_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASTE="$SCRIPT_DIR/taste.md"

source "$SCRIPT_DIR/env.sh"

echo "[meta-gardener] Prior domain: $PRIOR_DIR"
echo "[meta-gardener] New domain:   $NEW_DIR"
echo ""

# --- Check prior domain has results ---
if [ ! -f "$PRIOR_DIR/results.tsv" ] || [ ! -s "$PRIOR_DIR/results.tsv" ]; then
    echo "[meta-gardener] Warning: prior domain has no results.tsv — seeding from template only"
fi

# --- Build the meta-gardener prompt ---
PROMPT_FILE="/tmp/meta-gardener-prompt-$$.md"

cat > "$PROMPT_FILE" <<'HEADER'
You are the meta-gardener. Your job is to write the optimal program.md seed for a new research run, given everything learned from the prior run.

You are NOT running experiments. You are NOT the gardener. You do ONE thing:
read the prior run's artifacts, reason about what is known vs genuinely open,
and write a program.md that gives agents the best possible starting conditions.

## What makes a good program.md seed

**Seed what is known.** If the prior run proved that architecture X + hyperparameter Y works,
tell agents that. Don't make them rediscover it. Every experiment spent rediscovering known
results is wasted budget.

**Leave open what is genuinely unknown.** Don't over-specify. If the prior run never tested
the 200M sample budget with the mature architecture, that's an open question — say so explicitly.
Agents should explore, not just reproduce.

**Calibrate the baseline and ceiling honestly.** State what the prior run achieved and under
what conditions. Distinguish: "proven at 200M" vs "best at 50M" vs "theoretical ceiling."
Agents need accurate priors to make good decisions.

**Scope to the budget.** Agents have limited experiments. The init should match the gap
they can realistically close. A vanilla start with a large gap and a small budget is a
recipe for a stalled run.

**Name the open questions explicitly.** List the 2-3 highest-value unknowns the prior run
left unresolved. These are the most important things for agents to test first.

## Output format

Your entire response must be the program.md file content and nothing else.
No preamble. No explanation. No markdown code fences. No commentary after.
The very first character of your response must be the `#` of the title line.
If you add any explanation or meta-commentary, the file will be broken.

---

ARTIFACTS:

HEADER

# --- Prior run artifacts ---
echo "### taste.md (gardener principles)" >> "$PROMPT_FILE"
cat "$TASTE" >> "$PROMPT_FILE"
echo "" >> "$PROMPT_FILE"

if [ -f "$PRIOR_DIR/results.tsv" ] && [ -s "$PRIOR_DIR/results.tsv" ]; then
    echo "### prior results.tsv (top 30 by score)" >> "$PROMPT_FILE"
    sort -t$'\t' -k2 -rn "$PRIOR_DIR/results.tsv" | head -30 >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
    echo "### prior results.tsv (bottom 10 by score — failure modes)" >> "$PROMPT_FILE"
    sort -t$'\t' -k2 -n "$PRIOR_DIR/results.tsv" | head -10 >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

if [ -f "$PRIOR_DIR/meta-blackboard.md" ]; then
    echo "### prior meta-blackboard.md (distilled observations)" >> "$PROMPT_FILE"
    cat "$PRIOR_DIR/meta-blackboard.md" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

if [ -f "$PRIOR_DIR/blackboard.md" ]; then
    echo "### prior blackboard.md (last 100 lines)" >> "$PROMPT_FILE"
    tail -100 "$PRIOR_DIR/blackboard.md" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

if [ -f "$PRIOR_DIR/best/config.yaml" ]; then
    echo "### prior best/config.yaml" >> "$PROMPT_FILE"
    cat "$PRIOR_DIR/best/config.yaml" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

if [ -f "$PRIOR_DIR/best/sae.py" ]; then
    echo "### prior best/sae.py" >> "$PROMPT_FILE"
    cat "$PRIOR_DIR/best/sae.py" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

# --- New domain template program.md (if exists) ---
if [ -f "$NEW_DIR/program.md" ]; then
    echo "### new domain template program.md (rewrite this)" >> "$PROMPT_FILE"
    cat "$NEW_DIR/program.md" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

# --- Call Claude ---
echo "[meta-gardener] Calling Claude to seed program.md..."
SEED_FILE="$NEW_DIR/program.md.seed"

claude --dangerously-skip-permissions --max-turns 10 < "$PROMPT_FILE" > "$SEED_FILE"

LINES=$(wc -l < "$SEED_FILE" | tr -d ' ')
echo "[meta-gardener] Seed generated ($LINES lines)"

# Atomic replace
mv "$SEED_FILE" "$NEW_DIR/program.md"
echo "[meta-gardener] Wrote $NEW_DIR/program.md"
echo ""

# Clean up
rm -f "$PROMPT_FILE"

# --- Hand off to outer-loop ---
MAX_GENS="${3:-5}"
NUM_AGENTS="${4:-4}"
MAX_TURNS="${5:-200}"
MONITOR_MIN="${6:-20}"

echo "[meta-gardener] Handing off to outer-loop.sh..."
echo "[meta-gardener] Args: gens=$MAX_GENS agents=$NUM_AGENTS turns=$MAX_TURNS monitor=${MONITOR_MIN}m"
echo ""

exec bash "$SCRIPT_DIR/outer-loop.sh" "$NEW_DIR" "$MAX_GENS" "$NUM_AGENTS" "$MAX_TURNS" "$MONITOR_MIN"
