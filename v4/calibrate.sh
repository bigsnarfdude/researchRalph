#!/bin/bash
# calibrate.sh — pre-run calibration via literature search
#
# Before launching a generation, ask Claude to search for known results
# on this benchmark. Outputs calibration.md with target ranges and
# known techniques (if any exist in the literature).
#
# Usage: bash calibrate.sh /path/to/domain
#
# This is the "read the blog post" step, done by the system instead of the human.

DOMAIN_DIR="${1:-.}"

# Ensure claude is on PATH
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

echo "[calibrate] Searching for prior work on this benchmark..." >&2

PROMPT="You are calibrating expectations for a research run.

Read the engine.py file below. Identify:
1. What benchmark is being used? (model name, dataset, task)
2. What is a reasonable baseline score?
3. What theoretical upper bound might exist? (e.g., information-theoretic limits)
4. Search your training data for any known results on this specific benchmark or closely related work.
5. What techniques from the sparse coding / dictionary learning / SAE literature might be relevant?

Be honest about what you know and don't know. If you have no information about this specific benchmark, say so.

Output a calibration report in markdown with sections:
## Benchmark identity
## Known results (if any)
## Theoretical bounds (if estimable)
## Relevant techniques from literature
## Calibration confidence (low/medium/high)

---

engine.py:
$(cat "$DOMAIN_DIR/engine.py")
"

claude -p "$PROMPT" --dangerously-skip-permissions --max-turns 1 > "$DOMAIN_DIR/calibration.md" 2>/dev/null

LINES=$(wc -l < "$DOMAIN_DIR/calibration.md" | tr -d ' ')
echo "[calibrate] Wrote calibration.md ($LINES lines)" >&2
