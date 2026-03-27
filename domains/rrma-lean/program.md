# rrma-lean — Agent Instructions

## Task

Solve MiniF2F theorems in Lean 4. Each problem is a theorem statement with `sorry`.
Your job: replace `sorry` with a valid Lean 4 proof that compiles.

Score = fraction of MiniF2F valid problems solved.
Baseline: 0.0 (no attempts). Target: 0.50 (state of the art is ~0.45-0.60).

## How to run an experiment

```bash
# 1. Pick a set of problems to attempt
# 2. Write proof attempts to:
#    ~/researchRalph/domains/rrma-lean/attempts/<method_name>/<problem_name>.lean
# 3. Each attempt file must be a complete Lean 4 file:

# Example attempt file:
import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4 (x e : ℂ)
    (h₀ : x + e = 7) (h₁ : 2 * x + e = 3) : e = 11 ∧ x = -4 := by
  constructor <;> linarith [h₀, h₁]

# 4. Score it:
bash run.sh <method_name>
```

## Proof strategies to try

**Tactic-based (most problems):**
- `omega` — linear arithmetic over integers/naturals
- `linarith` — linear arithmetic over ordered fields
- `ring` — ring identities
- `norm_num` — numerical computations
- `simp` + `ring` — simplification + algebra
- `nlinarith` — nonlinear arithmetic
- `decide` — decidable propositions (small finite cases)
- `field_simp` + `ring` — field equations

**For harder problems:**
- `constructor` + sub-goals
- `intro` + `cases` + `simp`
- `have : ... := by ...` intermediate lemmas
- `calc` blocks for step-by-step equalities

## Problem categories in MiniF2F

- `algebra_*` — polynomial identities, equations (easiest, try first)
- `amc*` — AMC competition problems
- `aime_*` — AIME problems (harder)
- `mathd_*` — MATH dataset problems
- `imo_*` — IMO problems (hardest)

## Strategy

Start with algebra problems — high solve rate with `ring`, `linarith`, `omega`.
Build up to amc/mathd. Leave imo for last.

A method = a strategy for selecting which tactics to try and in what order.

## Scoring

```
score = problems_solved / problems_attempted
```

Only score problems you actually wrote attempts for.
A method that solves 50/100 attempts scores 0.50.

## Results protocol

Append to results.tsv:
```
EXP-ID  score  n_attempted  status  description  agent  design
```

- design: tactic (pure tactic), llm-guided, hybrid, search
- description: which problem categories + which tactics you prioritized

## TrustLoop rules

1. Proof must compile without sorry
2. Score on valid set only (not test)
3. No hardcoding answers — the proof must be general
