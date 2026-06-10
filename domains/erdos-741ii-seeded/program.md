# erdos-741ii-seeded

## Task
Prove `erdos_741_ii` in Lean 4 + Mathlib. See LEARNINGS.md for the full proof strategy.

The theorem: ∃ A : Set ℕ, A is an additive basis of order 2 AND for every partition
A = A₁ ∪ A₂ (disjoint), at least one of A₁+A₁ or A₂+A₂ is not syndetic.

## Key insight (in LEARNINGS.md)
Construction uses Pₖ = 100^k, forbidden zones Zₖ, oases xₖ = 10·Pₖ.
The pigeonhole argument forces unbounded gaps in one component's sumset.

## Oracle
bash run.sh — SCORE=1.0 on clean compile with 0 sorry.

## File to edit
workspace/agentN/Erdos741ii.lean — your private copy.

## Workflow
1. Read LEARNINGS.md carefully — the full proof is there in natural language
2. Translate it step by step into Lean
3. bash run.sh after each meaningful change
4. Share progress and key lemmas on blackboard.md
5. Build on what other agents write — don't duplicate work
