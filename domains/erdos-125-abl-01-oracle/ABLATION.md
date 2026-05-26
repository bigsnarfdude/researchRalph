# Ablation 01: Oracle Fix Removed (grep -c exits 1 on zero sorries)

**Removed:** `|| true` after `grep -c "sorry"` in run.sh

**Effect:** When sorry=0 (proof complete), grep -c returns exit code 1.
With set -e active, the script dies before logging SCORE=1.0. Agents receive
no feedback that they succeeded — the run silently produces 0 experiments.

**Prediction:** 0% SCORE=1.0 logged, regardless of proof quality.

**Confirms:** Oracle silence was catastrophic — this was the original failure mode
that burned 300+ Sonnet turns before the fix.
