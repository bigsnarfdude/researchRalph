# erdos-125-claudeception

## Task
Prove `erdos_125 : ∃ n : ℕ, n ∉ setAB` in Lean 4 + Mathlib.

setA = naturals whose base-3 digits are all 0 or 1
setB = naturals whose base-4 digits are all 0 or 1
setAB = {a + b | a ∈ setA, b ∈ setB}

## Oracle
bash run.sh — SCORE=1.0 on clean compile with 0 sorry.

## File to edit
workspace/agentN/Erdos125.lean — your private copy only.

## Workflow
1. Read Erdos125.lean and LEARNINGS.md
2. Edit workspace/agentN/Erdos125.lean
3. bash run.sh — check score and errors
4. Repeat until SCORE=1.0
