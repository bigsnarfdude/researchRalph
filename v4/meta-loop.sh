#!/bin/bash
# meta-loop.sh — the "sleep cycle" meta-agent
#
# Runs periodically during a research run.
# Reads blackboard + results + its own previous meta-blackboard.
# Compresses, prunes, reflects, writes updated meta-blackboard.md.
#
# This is the recursive self-improvement core:
#   meta-blackboard v1 -> agents explore -> meta-blackboard v2 (reads v1 + new results)
#   -> agents adjust -> meta-blackboard v3 (reads v2 + new results) -> ...
#
# Usage: bash meta-loop.sh /path/to/domain [interval_minutes]
#
# The meta-agent never directs. It observes, compresses, and reflects.
# Agents read meta-blackboard.md if they want. It's shared memory, not a boss.

DOMAIN_DIR="${1:-.}"
INTERVAL="${2:-30}"  # minutes between sleep cycles

# Ensure claude is on PATH
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"

echo "Meta-loop starting"
echo "  Domain: $DOMAIN_DIR"
echo "  Interval: ${INTERVAL}m"
echo ""

CYCLE=0

while true; do
    # Wait for enough data before first cycle
    RESULT_LINES=$(wc -l < "$DOMAIN_DIR/results.tsv" 2>/dev/null | tr -d ' ')
    BB_LINES=$(wc -l < "$DOMAIN_DIR/blackboard.md" 2>/dev/null | tr -d ' ')

    if [ "${RESULT_LINES:-0}" -lt 3 ] && [ "$CYCLE" -eq 0 ]; then
        echo "[meta] Waiting for data... (results: ${RESULT_LINES:-0} lines, blackboard: ${BB_LINES:-0} lines)"
        sleep 120
        continue
    fi

    CYCLE=$((CYCLE + 1))
    echo "[meta] === Sleep cycle $CYCLE starting at $(date '+%H:%M:%S') ==="

    # Build the prompt
    PROMPT_FILE="/tmp/meta-loop-prompt-$$.md"

    cat > "$PROMPT_FILE" <<'HEADER'
You are the meta-agent in a recursive research loop. Your job is to OBSERVE, COMPRESS, and REFLECT — never to direct.

Below are the current artifacts from an ongoing multi-agent research run, plus your own previous meta-blackboard (if any).

Write an updated meta-blackboard.md. It must have these sections:

## Current best
The best score achieved so far and the config/architecture that produced it.

## What works (ranked by impact)
Techniques that improved scores. Include approximate gains and WHY each works.

## Dead ends
Approaches that failed. Include scores. Group by category. This is the PRUNING function — agents should skip these.

## Patterns noticed
What you observe about the PROCESS, not just results. Examples:
- "Three agents are all sweeping lr — this is saturated"
- "Nobody has tried the blind spots from last cycle"
- "Scores plateaued at 0.98 for the last 5 experiments"
- "Agent2's crossover approach from cycle N is promising but wasn't followed up"

## Blind spots
Approaches never tried. Update from previous cycle — remove any that were tried, add new ones you notice.

## Stepping stones
Ideas that didn't beat the best score but produced interesting intermediate results worth building on. These are the non-obvious paths that might matter later.

## Surprises
What happened this cycle that contradicted prior expectations? List each as:
- Expected: [what was predicted]
- Actual: [what happened]
- Why the gap existed: [what assumption was wrong]

Surprises are calibration failures — the most information-dense events in a run.

## Devil's advocate
Make the strongest case that the current best score is WRONG, inflated, or misleading.
Consider: metric gaming, sorry/shortcut proofs, oracle quirks, evaluation leakage,
results that wouldn't generalize beyond this benchmark. Be specific.
If the score is genuinely solid, say so and explain why.

## Self-reflection
Compare this meta-blackboard to your previous one (if any). What did you recommend last time? Did agents follow it? Did it help? What should you say differently this time?

Rules:
- Be concise. Target 120 lines max.
- State confidence levels.
- Do NOT give instructions or tell agents what to do. Describe what you SEE.
- Focus on saving experiments. Every line should prevent a wasted run.
- If this is your first cycle, skip the self-reflection and devil's advocate sections.

---

ARTIFACTS:

HEADER

    # Include previous meta-blackboard if it exists
    if [ -f "$DOMAIN_DIR/meta-blackboard.md" ]; then
        echo "### Previous meta-blackboard.md (cycle $((CYCLE - 1)))" >> "$PROMPT_FILE"
        cat "$DOMAIN_DIR/meta-blackboard.md" >> "$PROMPT_FILE"
        echo "" >> "$PROMPT_FILE"
        echo "---" >> "$PROMPT_FILE"
        echo "" >> "$PROMPT_FILE"
    fi

    echo "### blackboard.md (last 200 lines)" >> "$PROMPT_FILE"
    tail -200 "$DOMAIN_DIR/blackboard.md" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"

    echo "### results.tsv" >> "$PROMPT_FILE"
    cat "$DOMAIN_DIR/results.tsv" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"

    echo "### best/config.yaml" >> "$PROMPT_FILE"
    cat "$DOMAIN_DIR/best/config.yaml" >> "$PROMPT_FILE"
    echo "" >> "$PROMPT_FILE"

    if [ -f "$DOMAIN_DIR/best/sae.py" ]; then
        echo "### best/sae.py" >> "$PROMPT_FILE"
        cat "$DOMAIN_DIR/best/sae.py" >> "$PROMPT_FILE"
        echo "" >> "$PROMPT_FILE"
    fi

    # Run through Claude — pipe via stdin to avoid ARG_MAX limits on large prompts
    claude --dangerously-skip-permissions --max-turns 3 < "$PROMPT_FILE" > "$DOMAIN_DIR/meta-blackboard.md.tmp"

    # Atomic replace
    mv "$DOMAIN_DIR/meta-blackboard.md.tmp" "$DOMAIN_DIR/meta-blackboard.md"

    LINES=$(wc -l < "$DOMAIN_DIR/meta-blackboard.md" | tr -d ' ')
    echo "[meta] Wrote meta-blackboard.md ($LINES lines)"
    echo "[meta] Sleeping ${INTERVAL}m until next cycle..."
    echo ""

    # Clean up
    rm -f "$PROMPT_FILE"

    sleep $((INTERVAL * 60))
done
