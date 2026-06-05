# Stoplight — erdos-741ii-g2-opus
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 4 | Stagnation: 3 since last breakthrough

## Dead ends — do NOT retry
- Design '' has 4 experiments, 0 keeps — abandon this approach

## Agents
- agent0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- agent1: 1 exp, 0 breakthroughs, rate 0%, best 1.0
- agent2: 1 exp, 0 breakthroughs, rate 0%, best 1.0
- agent3: 1 exp, 1 breakthroughs, rate 0%, best 0.0

## Recent blackboard (last 20 entries)
**Oracle:** `bash run.sh` — returns SCORE=1.0 when workspace file compiles with 0 sorry.
## Problem
Erdős #741(ii): Exhibit A ⊆ ℕ that is an additive basis of order 2, such that for every
partition A = A₁ ⊔ A₂, at least one of (A₁+A₁), (A₂+A₂) has unbounded gaps.
## Scaffold
G2 — skeleton with 2 sorries. Fill them both to win.
## Sorries to fill
1. `basis_lem` — coverage: [4,6·5^k] ⊆ Akn(k+1)+Akn(k+1)
2. `rigidity` — Jk zone forces ck+Bk pairs
## Status
[x] PROVED by agent1 and agent3
## agent1 — Attempt 1 — SCORE=1.0
Strategy: Read calibration.md → followed Step 0 (read g3 reference proof lines 78–292) → added 3 helper lemmas (singleton_add_icc, icc_add_singleton, pair_eq') → filled basis_lem (induction, 13 by_cases thresholds) → filled rigidity (lt_trichotomy, rw [hje] at haj for j=k branch).
Result: SORRY_COUNT=0, BUILD_EXIT=0, SCORE=1.0 in first oracle call.
Key tactics that worked:
- basis_lem inductive step: new_cov covers [4Qk,30Qk] with explicit pairs; ih' handles [4,6Qk] via akn_mono
- rigidity j=k branch: rw [hje] at haj (NOT subst); omega for Nat subtraction; linarith for linear bounds
## agent3 — Attempt 1 — SCORE=1.0
Same strategy as agent1. SORRY_COUNT=0, BUILD_EXIT=0, SCORE=1.0 on first oracle call. Blueprint in calibration.md is complete and reproducible.
