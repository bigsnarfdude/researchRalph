# MISTAKES.md

## M1: tol=1e-12 crashes nontrivial branches at n=300 (agent0)
What: Set solver_tol=1e-12 with n_nodes=300 on positive/negative branches
Result: CRASH — solver fails to converge
Lesson: Nontrivial solutions need more DOF per unit tolerance. tol=1e-11 is the practical limit at n=300.

## M2: Initial sweep was too coarse (agent0)
What: Swept u_offset in steps of 0.1 from 0.0 to 1.0
Result: Missed the fractal boundary structure between 0.52-0.58
Lesson: Near bifurcation points, use finer resolution (0.01 steps) to capture basin structure.

## M3: Gen2 stopped at n=195 without testing 196 (agent0, gen3)
What: Previous generation found n=195 was best, declared it optimal
Result: n=196 is actually 1.3% better (1.47e-12 vs 1.49e-12)
Lesson: When the residual curve is jagged, always test ALL integers in the promising range. Don't stop at the first local minimum.
