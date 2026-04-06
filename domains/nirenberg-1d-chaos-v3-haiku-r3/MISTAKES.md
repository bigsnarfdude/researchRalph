# Agent2 Mistakes & Corrections

## Assumption: u_offset Fine-Tuning Would Improve Positive Branch

**Mistake:** Tested u_offset=0.95 with expectation of better convergence than u_offset=0.9.

**Result:** exp016 (u_offset=0.95) yielded identical residual to exp012 (u_offset=0.9): both 8.82e-11.

**Root cause:** Once solver converges to the correct branch, u_offset variation within the basin of attraction (0.9–0.95) produces the same solution. Branch selection is "coarse" — only gross u_offset changes (e.g., 0→0.9→−0.9) matter.

**Lesson:** Time better spent on solver parameters (mesh, tolerance) than on initial condition offset fine-tuning. Solver parameters directly drive residual quality; u_offset selection is binary (which branch) not continuous (what residual).

**Action taken:** Pivoted to solver-parameter focus. No more u_offset experiments unless exploring new branches.
## Agent1 Mistakes / Failed Attempts

1. **Exp050: tol=1e-12 crashed solver**
   - Expected tighter tolerance to help; instead caused numerical instability
   - Lesson: there's a sweet spot around 1e-11; beyond that the solver diverges

2. **Assumption of symmetry in bifurcation**
   - Initially expected negative and positive boundaries to be symmetric
   - Reality: asymmetric boundaries (-0.625 vs +0.57) due to K(θ) breaking mirror symmetry
   - Lesson: don't assume ideal properties; measure everything

3. **Over-focus on modes and amplitudes early on**
   - Spent several experiments on mode variations (1, 2, 3)
   - Modes don't significantly affect branch selection or residual quality
   - Lesson: identify the key leverage point first (u_offset), optimize others later
