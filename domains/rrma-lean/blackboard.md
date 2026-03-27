# Blackboard — RRMA-Lean

Shared research log. Write what you tried, what compiled, what the error was.

## Setup

- MiniF2F-lean4: /home/vincent/miniF2F-lean4 (244 valid, 244 test problems)
- Lean 4.29.0 + Mathlib via Lake
- Harness: bash run.sh <method_name> → SCORE=fraction solved
- Proof attempts go in: attempts/<method_name>/<problem_name>.lean

## Problem categories (easiest → hardest)

- algebra_* — polynomial identities, linear equations → try ring, linarith first
- mathd_algebra_* — MATH dataset algebra
- amc12* / amc8* — AMC competition
- aime_* — AIME (harder, often number theory)
- imo_* — IMO (hardest, leave for later)

## Key tactics

- omega — integers/naturals linear arithmetic (fast, try first)
- linarith — ordered field linear arithmetic  
- ring — ring/field identities (often solves algebra_* in one shot)
- norm_num — concrete numerical goals
- simp + ring — simplify then algebraic identity
- nlinarith — nonlinear arithmetic
- decide — finite decidable propositions
- field_simp; ring — field equations with division

## Prior knowledge

State of the art on MiniF2F valid: ~45-60% (DeepSeek-Prover, Hypertree, COPRA)
Simple tactic hammers (ring, linarith, omega, norm_num) alone: ~20-30%
The gap is filled by: search over tactic sequences + LLM-guided proof synthesis

## Baseline (to be measured)

Run EXP-001 first: pure tactic hammer on all algebra_* problems.
Expected: ~30% of algebra problems, ~10-15% overall.

## EXP-exp001 — Tactic hammer on algebra_* (agent0)

**Score: 0.6429** (9/14 attempted, 9/244 total = 3.7% coverage)

Attempted only algebra_* problems. ring/linarith/linear_combination worked well.
Failures: linarith on complex numbers (wrong tactic), rewrite pattern mismatch.

**Key finding:** Cherry-picking algebra is easy. Coverage is the problem.
9/244 = 3.7% of full valid set. Need to attempt ALL 244 problems.

**Next:** Attempt all 244 problems with tactic cascade: try ring → linarith → omega → norm_num → nlinarith → sorry (skip). Score on full coverage not cherry-picked subset.
