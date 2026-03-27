#!/bin/bash
# calibrate.sh — pre-run literature search via web + HF papers
#
# Searches for current SOTA, known techniques, and relevant papers
# before launching a generation. Writes calibration.md.
#
# Usage: bash calibrate.sh /path/to/domain

DOMAIN_DIR="${1:-.}"

source "$(cd "$(dirname "$0")" && pwd)/env.sh"

echo "[calibrate] Searching literature (web + HF papers)..." >&2

DOMAIN_NAME=$(basename "$DOMAIN_DIR")
PROGRAM=$(cat "$DOMAIN_DIR/program.md" 2>/dev/null | head -40)

PROMPT="You are calibrating a research run for domain: $DOMAIN_NAME

Read program.md excerpt below to understand the task, then do the following research:

1. Use WebSearch to find current SOTA results on this benchmark (search for the benchmark name + 'SOTA 2024 2025 results leaderboard')
2. Use WebSearch to find recent papers on the specific approach (search for key methods + 'arxiv 2024 2025')
3. Use WebSearch to search HuggingFace papers: site:huggingface.co/papers for relevant recent work
4. Search for known failure modes and what NOT to try

For rrma-lean / MiniF2F specifically, search for:
- 'MiniF2F Lean 4 SOTA 2024 2025'
- 'DeepSeek-Prover MiniF2F results'
- 'Goedel Code Prover tactics'
- 'Lean 4 tactic proof search nlinarith omega linarith'
- 'COPRA Lean proof search'
- 'Hypertree proof search MiniF2F'

Output calibration.md with:
## Benchmark identity
## Current SOTA (with numbers and citations)
## Best known techniques (specific tactics, strategies, approaches)
## What has been tried and failed
## Recommended starting point for this run
## Sources searched

Be specific — actual numbers, actual tactic names, actual paper titles.

---
program.md (first 40 lines):
$PROGRAM
"

claude -p "$PROMPT" \
    --dangerously-skip-permissions \
    --max-turns 5 \
    > "$DOMAIN_DIR/calibration.md" 2>/dev/null

LINES=$(wc -l < "$DOMAIN_DIR/calibration.md" | tr -d ' ')
echo "[calibrate] Wrote calibration.md ($LINES lines)" >&2
