# erdos-125-g0-n8 — cold start, no scaffold

## Task
Prove `erdos_125 : ∃ n : ℕ, n ∉ setAB` in Lean 4 + Mathlib.

setA = naturals whose base-3 digits are all 0 or 1
setB = naturals whose base-4 digits are all 0 or 1
setAB = {a + b | a ∈ setA, b ∈ setB}

Prove there exists a natural number not in setAB.

## Oracle
bash run.sh — compiles your workspace file, returns SCORE=1.0 on clean compile with 0 sorry.

## File to edit
workspace/agentN/Erdos125.lean — your private copy. Edit only this file.

## Workflow
1. Read Erdos125.lean — understand what needs to be proved
2. Edit workspace/agentN/Erdos125.lean — replace sorry with a proof
3. bash run.sh — see score and any compiler errors
4. Repeat until SCORE=1.0
