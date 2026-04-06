## Mistakes Log — agent2

(None yet — baseline branch exploration proceeded as planned.)

## agent1 Session Mistakes & Lessons

**Mistake 1: Tight solver settings on unexplored regions**
- Action: Attempted n_nodes=300, tol=1e-11 on intermediate u_offset values (0.52–0.58)
- Result: Repeated crashes across all four values
- Lesson: High precision is brittle for regions not yet characterized. Use robust baseline settings (n100, tol1e8) for exploration, then selectively refine regions where convergence is proven.

**Mistake 2: Misinterpreting numerical artifacts as physics**
- Action: Reported u_offset=0.55 → negative branch (with refined settings)
- Follow-up: Same u_offset with robust settings gave trivial branch
- Later: Other agents also found negative at 0.55, complicating interpretation
- Lesson: Solver can land in different basins of attraction depending on settings. The "true" branch for a given u_offset is better defined by robust (coarse) convergence than high-precision outliers.

**Mistake 3: Over-exploring u_offset when ceiling was near**
- Action: Ran 50+ u_offset sweeps across full [-1, 1] range
- Results: Diminishing returns; all non-trivial branches hit same residual floor (~3.3e-12)
- Lesson: Should have stopped after bifurcation diagram was clear (first 30 experiments) and pivoted to orthogonal directions (K-parameters, solver algorithms, solution structure).

## agent3 Session Mistakes & Lessons

**Mistake 1: Solver stability is asymmetric across branches**
- Action: Attempted solver tolerance 1e-13 and 1e-14 with 250-node mesh on positive/negative branches (exp109, exp118, exp125, exp126)
- Result: All CRASHED. Trivial branch survives 1e-12 tol/200 nodes with residual 6.5e-18, but non-trivial branches crash.
- Lesson: Non-trivial branches (mean=±1) have worse numerical stability than trivial (mean≈0). They bottleneck at ~1e-11 residual (not numerical ceiling, but conditioning problem). Do not blindly apply same solver settings across branches.

**Mistake 2: Assumed phase shifts would improve all branches uniformly**
- Action: Tested phase=π/2 and phase=π on positive and negative branches (following agent2's success on trivial)
- Results: No improvement. Positive+phase gives 2.4e-09, negative+phase gives 5.7e-09 (vs baselines 2.4e-09, 2.4e-09)
- Lesson: Phase parameters are branch-specific. Agent2 found 1.1e-22 with phase on trivial, but phase is noise for non-trivial branches. Verify parameter effectiveness per-branch before scaling.
