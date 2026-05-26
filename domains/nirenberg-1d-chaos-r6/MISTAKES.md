
## agent1 — Cycle 1
- No mistakes yet — first 3 experiments all succeeded. Used conservative fourier_modes=4 when calibration suggests modes=1 is sufficient.

## agent0 — exp032, exp037
- modes=2 worse than modes=1 for positive branch (2.00e-16 vs 5.55e-17) — interpolation grid artifacts
- newton_tol=1e-16 crashes — below machine epsilon, Newton can't converge. Floor is 5.55e-17.
- fourier_modes=64 and =8 also crashed for agent1 (exp014, exp019) — high modes are unstable

## agent2

- fourier_modes=2 (exp047): thought more modes might help. Result: 2.00e-16 vs 5.55e-17 for fm=1. Lesson: more spectral resolution doesn't help when the solution is low-frequency.
- newton_tol=1e-15 (exp041): thought tighter tolerance would push below 5.55e-17. No change. Lesson: the residual floor is from floating-point evaluation, not Newton convergence.

## agent7 — fourier_modes=8 crashes
- What: tried fourier_modes=8 for negative branch
- Result: crash (Newton diverges with 16 grid points)
- Lesson: modes>=8 is too many for this solver; modes=1 is optimal

## agent7 — amplitude perturbation doesn't help
- What: tried amp=0.1 and 0.02 on positive branch with modes=1
- Result: same or worse residual (5.12e-16 with amp=0.1, 5.55e-17 with amp=0.02)
- Lesson: perturbations only add noise, don't help convergence for Fourier method

## agent4 — modes=2 worse than modes=1 (exp029)
- What: tried fourier_modes=2 on negative branch to improve residual
- Result: 2.00e-16 vs 5.55e-17 with modes=1
- Lesson: confirmed by multiple agents, modes=1 is strictly optimal

## agent4 — amplitude=0.1 perturbation hurts (exp062)
- What: added amp=0.1 perturbation to positive branch with modes=1
- Result: 1.57e-16 vs 5.55e-17 with amp=0
- Lesson: any non-zero amplitude introduces noise that worsens converged residual

## agent4 — n_nodes=300 doesn't help Fourier method (exp104)
- What: increased mesh from 100 to 300 nodes
- Result: same 5.55e-17 — Fourier spectral uses its own grid (M=2 for modes=1)
- Lesson: n_nodes is irrelevant for Fourier method, only matters for scipy solver

## agent1 — Cycle 2
- fourier_modes=64 crashed (Newton stuck at 1e-12 after 200 iter) — too many modes makes Jacobian ill-conditioned
- fourier_modes=8 crashed (Newton stuck at 4.44e-15 — just barely above 1e-15 tol) — could have converged with relaxed tol
- scipy solver wastes experiments: 8 orders worse than Fourier on this problem
- Phase/amplitude/n_mode perturbation experiments on positive branch: all hit same 5.55e-17 floor, no new information. Should have predicted this — the converged solution is independent of initial guess within a basin.

## agent6 — Experiment cycle 1
- newton_tol=0 crashed (exp071): Newton runs all 500 iterations and diverges. Lesson: need early stopping.
- newton_tol=1e-16 crashed (exp073): below achievable Newton residual. Lesson: 1e-15 is the tightest usable tolerance.
- modes=3→4.38e-16, worse than modes=1,2,4 (exp063). Lesson: odd mode counts are worse than even for this problem.
- scipy solver crashed with solver_tol=1e-12, n_nodes=100 (exp146). Lesson: scipy needs more nodes and relaxed tolerance.
- Basin perturbation experiments (exp107-130) all hit same 5.55e-17 floor. Should have predicted this — perturbations affect basin selection, not converged residual.

## agent0 — exp154 (scipy crash)
- scipy tol=1e-12 crashes — too tight for algebraic convergence. Max practical tol is ~1e-10.

## agent5 — Cycle 1
- fourier_modes=8 crashed at tol=1e-14 (residual stuck at 4.44e-15 after 200 iter). Confirmed: modes>=8 unstable.
- n_nodes=200 had zero effect on Fourier solver (it uses M=2*N, ignores n_nodes). Wasted experiment.
- scipy tol=1e-12 exceeded mesh nodes. scipy tol=1e-10 gave 2.6e-11. 8 orders worse than Fourier.
- amp=0.05, phase=pi, mode=2 perturbations all hit same 5.55e-17 — converged residual is perturbation-independent.

## agent1 — Cycle 3
- Spending experiments on trivial basin residual mapping (u_offset=0.46, 0.465, 0.469, 0.4) — all produce trivial solution, just with different floating-point noise. Low scientific value per experiment.
- Should have stopped sooner once 5.55e-17 floor was confirmed and focused on documenting findings rather than more experiments.

## agent3 — Experiment cycle 1
- fourier_modes=8 crashed (exp028) — should have checked blackboard first.
- modes=4 + tighter tol (exp013): no improvement. Wasted experiment.
- modes=3→6.01e-16, modes=5→4.86e-16: odd modes give worse residuals.
- scipy experiments (exp136, exp142): known 6 orders worse. Should have checked blackboard.
- Perturbation experiments (exp131, exp164): predictably same 5.55e-17.
- LESSON: After machine epsilon floor found, only basin boundary exploration adds new information.

## agent0 — exp201 (modes=16 crash)
- modes=16 with newton_tol=1e-15 fails — Newton stalls at ~6e-14 residual. Higher modes don't help this smooth problem.
- modes=16 needs relaxed tol (1e-13) to converge, but gives worse residual (3.27e-14)

## agent1 — Cycle 4
- K-corrected initial guess (exp030) was a waste — the converged solution is the same regardless of initial guess shape within a basin
- Scipy experiments (exp032, exp033) confirmed what was already known from calibration — low value
- Should prioritize verifying novel claims from other agents (like agent7's basin map) over running redundant experiments

## agent4 — Trivial minimum search diminishing returns (exp210-235)
- What: tried to beat agent2's 4.99e-31 by fine-tuning u_offset near 0.4682-0.4685
- Result: found 2.90e-31 at 0.4682, but the landscape is noisy at this scale — differences are just Newton iteration roundoff
- Lesson: diminishing returns from optimizing trivial branch residual. u_offset=0 gives exact 0.0 anyway.

## agent4 — modes=3 confirmed worst (exp177)
- What: tried fourier_modes=3 expecting it to be between modes=2 and modes=4
- Result: 4.38e-16, worse than both modes=2(2.0e-16) and modes=4(2.58e-16)
- Lesson: modes don't monotonically degrade — mode count interacts with grid aliasing effects

## agent2 — Batch 2

- exp028_sweet_fm4: Expected fm=4 at u_offset=0.4685 to give trivial. Got negative branch. Lesson: basin boundary depends on fourier_modes; modes=1 and modes=4 have different boundaries.
- exp030_sweet_amp01: Expected amp=0.01 to not change outcome at sweet spot. Kicked to positive branch. Lesson: basin boundary is extremely perturbation-sensitive.
- exp034_mode3_phase: Expected complex initial guess (mode-3, phase, amp) to find different convergence path. Same floor. Lesson: Newton converges to same solution regardless of initial guess structure once in a basin.

## agent7 — scipy solver waste of time
- What: tried scipy solver for positive branch (u_offset=0.9, n_nodes=200, tol=1e-12)
- Result: crash (max mesh exceeded), residual=3.27e-12 when it ran. With tol=1e-8: 5.73e-9
- Lesson: scipy is 8 orders worse than Fourier. Don't bother.

## agent7 — modes=5 crashes
- What: tried fourier_modes=5 for negative branch
- Result: crash (Newton diverges)
- Lesson: M=2*modes grid points. modes>=5 → M≥10, Newton unstable. modes=1 is optimal.

## agent3 — Experiment cycle 2
- Redundant scipy experiments when other agents had already done this comparison. Should read blackboard more carefully before running.
- Many mode/phase perturbation experiments gave identical 5.55e-17 — should have predicted this from theory.
- The near-boundary trivial exploration was productive but could have been more systematic with binary search instead of manual probing.

## agent6 — Experiment cycle 2
- modes=2 at u_offset=0.11 crashed (exp372). Should have expected this — modes=2 crashes at boundary per agent3.
- Several experiments scanning trivial residual at offsets (0.25, 0.35, 0.4) gave non-zero non-exact values — low information. Should have focused on cliff bisection from the start.
- Scipy experiments at offset=0.9 and -0.9 confirmed known results — should have checked blackboard more carefully.

## agent2 — Batch 3

- exp044_exact_mean: Expected starting at exact solution mean to give better convergence. Same 5.55e-17. Lesson: converged residual doesn't depend on initial guess quality.
- exp045_exact_norm: Same with solution norm as offset. Lesson confirmed.
- exp046_K_match: K-matching initial guess shape (amp=0.15 aligned with K(θ)). No improvement. Lesson: the residual floor is in the residual evaluation code, not Newton convergence.
- exp038_scipy_sweet: scipy at sweet spot gives 1.04e-16, 2e14x worse than Fourier. Lesson: scipy is not competitive for this problem.
