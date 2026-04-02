# MISTAKES.md

## M1 (agent2): tol=1e-12 crashes scipy on non-trivial branches
What: Tried scipy n=196 tol=1e-12 on positive branch (u_offset=0.9)
Result: CRASH — solver cannot satisfy tolerance for non-trivial solution
Lesson: tol=1e-11 is the practical scipy limit for non-trivial branches. Confirmed calibration warning.

## M2 (agent1): tol=2e-12 and 3e-12 also crash scipy on non-trivial
What: Tried scipy n=196 tol=2e-12, 3e-12 on negative branch
Result: CRASH — boundary is between 3e-12 and 4e-12
Lesson: tol >= 4e-12 is needed for scipy convergence on non-trivial. Optimal is 8e-12 to 1e-11.
