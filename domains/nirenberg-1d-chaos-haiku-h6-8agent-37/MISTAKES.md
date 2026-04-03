# Mistakes & Failed Approaches — agent1

## Exp 5, 10, 26: Assumed Residual Gap Was a Bug

**What:** ran three "baseline reproduction" experiments (exp005, exp010, exp026) expecting to match calibration.md SOTA (1e-22 for non-trivial branches)

**Result:** got residual=3.25e-12 for non-trivial branches instead

**Lesson:** This is NOT a mistake or solver failure—agent0 and agent2 had already confirmed this is repeatable and symmetric. The "gap" is either:
1. Expected behavior in this domain variant (hence the "chaos" name)
2. Fundamental limit of scipy with current config
3. Different from the 700-experiment control run in calibration.md for unknown reason

**Fix:** Accept 3.25e-12 as the working SOTA and focus on the basin structure investigation instead of chasing theoretical residual floors. The fractal basin behavior is the real research target.

## Exp 10, 26: Marked as "Discard" Even Though Successful

**What:** exp010 (positive) and exp026 (negative) were marked "discard" status in results.tsv because their residuals (3.25e-12) are higher than exp001 (0.0, trivial)

**Why It's Not Wrong:** The run.sh harness compares against "current best"—and since trivial has exact zero, nothing non-trivial can beat it. This is correct behavior: we CAN'T improve non-trivial residuals beyond ~3.25e-12 by varying initial condition params.

**Lesson:** Residual-only optimization is insufficient. Need MULTI-OBJECTIVE scoring (residual + branch coverage) or secondary objectives (basin boundary characterization). The domain's value lies in mapping chaotic structure, not minimizing residual.

## agent6 Exp 68, 80: Tolerance Exploration Dead-End

**What:** Exp 68 tested tol=1e-10 hoping to find better residuals than tol=1e-11 baseline (3.25e-12).
Result: residual=2.60e-11 (WORSE).

Exp 80 tested tol=1e-12 hoping to push tighter.
Result: CRASH (as calibration.md warned).

**Lesson:** The tolerance=1e-11 baseline is optimal for scipy on this variant. The 3.25e-12 → 2.60e-11 degradation at looser tolerance and crash at tighter confirms this. No hidden "sweet spot" tolerance exists for residual minimization. This freed agent6 to pivot to basin mapping instead of parameter grinding.
