#!/bin/bash
# generate-meta-blackboard.sh — post-run final distillation
#
# Run this AFTER a run completes (or between generations).
# Uses the same prompt as meta-loop.sh but is a one-shot for final compression.
#
# Usage: bash generate-meta-blackboard.sh /path/to/domain

DOMAIN_DIR="${1:-.}"

# Ensure claude is on PATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/meta_artifacts.sh"

if [ ! -f "$DOMAIN_DIR/blackboard.md" ]; then
    echo "Error: no blackboard.md found in $DOMAIN_DIR"
    exit 1
fi

echo "Generating final meta-blackboard..."

PROMPT_FILE="/tmp/meta-final-prompt-$$.md"

cat > "$PROMPT_FILE" <<'HEADER'
You just finished reading the complete artifacts from a multi-agent research run. Write a final cheat sheet for agents starting the same benchmark from scratch. This will be placed as meta-blackboard.md before they begin.

Include these sections:

## Winning recipe
The exact config that achieved the best score. Agents should validate this first.

## What works (ranked by impact)
Each technique that improved the score, with approximate gain and WHY it works.

## Dead ends
Every approach that failed. Include scores. Group by category.

## Scaling laws
Validated relationships between hyperparameters. Present as tables.

## Stepping stones
Intermediate results that didn't win but produced interesting properties worth exploring.

## Blind spots
Approaches never tried. Most promising directions for new work.

## Key insight
The single most important finding, in 2-3 sentences.

## Surprises
What happened that contradicted prior expectations? List each surprise as:
- Expected: [what the system/agents predicted]
- Actual: [what happened]
- Why the gap existed: [what assumption was wrong]

These are the most information-dense events in the run. Every surprise is a calibration failure worth preserving.

## Devil's advocate
Make the strongest case that the current best score is WRONG, inflated, or misleading.
Consider: metric gaming, evaluation leakage, sorry/shortcut proofs, overfitting to oracle quirks,
approaches that look good but wouldn't generalize. Be specific and ruthless.
If the score is genuinely solid, say so and explain why.

## Experiment order
What agents should do first, second, third with this cheat sheet.

Rules:
- Be concise. Target 150 lines max.
- State confidence levels.
- Focus on saving experiments.

---

ARTIFACTS:

HEADER

# Include previous meta-blackboard if it exists (for cross-generation recursion)
if [ -f "$DOMAIN_DIR/meta-blackboard.md" ]; then
    echo "### Previous meta-blackboard.md (from during the run)" >> "$PROMPT_FILE"
    cat "$DOMAIN_DIR/meta-blackboard.md" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
    echo "---" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"
fi

# v4.10: artifact assembly + validated write moved to meta_artifacts.sh, shared
# with meta-loop.sh. Previously this redirected straight onto meta-blackboard.md,
# so a failed model call destroyed the only cross-generation memory the run had.
append_domain_artifacts "$DOMAIN_DIR" "$PROMPT_FILE"

# Pipe via stdin, not argv — large blackboards + results.tsv blow past ARG_MAX.
TMP_OUT="$DOMAIN_DIR/meta-blackboard.md.tmp.$$"
claude --dangerously-skip-permissions --max-turns 3 < "$PROMPT_FILE" > "$TMP_OUT"
CLAUDE_EXIT=$?

rm -f "$PROMPT_FILE"

commit_meta_blackboard "$DOMAIN_DIR" "$TMP_OUT" "$CLAUDE_EXIT"
