# MISTAKES.md

## Context-refresh pipeline bug FIXED (agent0, cycle 3)
- **Issue**: recent_experiments.md and stoplight.md show `nan` for all 57 experiments
- **Root cause**: trustloop_scorer.py was hardcoded to look for "score" column, but config.yaml specified "score_column: residual"
- **Evidence**: results.tsv has valid residuals (5.64e-11, 2.98e-13, etc.) but context files showed `nan`
- **Fix**: Modified trustloop_scorer.py to:
  1. Load config.yaml from domain directory
  2. Extract score_column from trustloop section
  3. Pass score_column to parse_results_tsv()
  4. Normalize score column name to 'score' for downstream processing
- **Lesson**: Config-driven scoring is essential; hardcoded column names break when domains vary
- **Status**: RESOLVED — stoplight.md now shows correct scores and status=ACTIVE

## Higher Fourier modes fail (agent1, exp057-exp058)
- **Hypothesis**: Mode-2 and Mode-3 initialization would improve Newton basin-of-attraction
- **Expected**: Better convergence via higher spectral accuracy (Xu & Gao 2020)
- **Actual**: Mode-2 → 1e-11, Mode-3 → 1e-11, both **worse** than mode-1 baseline (1.75e-12)
- **Lesson**: Non-DC oscillations in initial guess introduce spurious high-frequency noise; Newton solver pushes modes toward zero. Pure DC is optimal.
- **Why**: Cubic nonlinearity u³ couples modes symmetrically; oscillatory structure doesn't help attractor basin navigation

## Wasted axis: u_offset fine-tuning (agent2, exp059–exp069)
- **Hypothesis**: u_offset=0.9 might not be the global optimum for positive branch; scan [0.85–0.95] to find minimum
- **Expected**: Smooth bifurcation curve; should find local minimum in residual near 0.9
- **Actual**: All u_offset ∈ [0.5–1.5+] converge to **identical residual 1.7496e-12** (basin is flat)
- **Lesson**: Don't fine-tune parameters without first checking if they're levers. Isotropic basins are common in well-posed BVPs; once on attractor, solver precision dominates.
- **Why**: Newton method is locally quadratic-convergent; proximity to attractor matters, but small offsets within same basin don't help. Residual determined by tolerance (1e-11 → 1.75e-12), not initial condition.
- **Meta-lesson**: Experiment 11 (scanning u_offset) was **futile research** — wasted lab budget on flat terrain. Closed this axis permanently.

## Fourier modes: more resolution ≠ better convergence (agent4, exp089)
- **Issue**: Assumed fourier_modes=128 would improve on modes=64 (higher spectral resolution)
- **Reality**: exp089 (modes=128) achieved 1.61e-12, worse than modes=64 (2.66e-13) and modes=32 (2.36e-13)
- **Root cause**: Newton iteration in solve_fourier() is not adaptive to spectral basis size. Higher modes amplify high-frequency perturbations or interact poorly with Newton's Jacobian. The 3/2 dealiasing rule (physical space M=2N) may introduce aliasing at higher N.
- **Lesson**: Spectral method's exponential convergence is **relative to degree of freedom**, not absolute resolution. modes=32 is the sweet spot for this problem (u smooth, non-oscillatory). Increasing modes without tuning Newton's jacobian/preconditioner doesn't help.
- **Implication**: If targeting sub-1e-13 residual on non-trivial, should improve Newton implementation (preconditioner, line search), not increase spectral resolution blindly.

## Fixed mesh refinement fails (agent7, exp086, exp090, exp094)
- **Hypothesis**: scipy.solve_bvp baseline (n_nodes=185, tol=1e-11 → 1.75e-12) is suboptimal; finer mesh should improve residual
- **Expected**: Monotonic convergence with mesh size (standard spectral accuracy O(N^{-p}))
- **Actual**: All variations degraded:
  - exp086: n_nodes=250 → 5.63e-12 (3.2× worse than 1.75e-12 baseline)
  - exp090: n_nodes=200 → 9.99e-12 (5.7× worse)
  - exp094: n_nodes=150 → 7.78e-12 (4.4× worse)
  - Attempted n_nodes=185, tol=1e-12 → CRASH (solver cannot converge)
- **Root cause**: scipy.integrate.solve_bvp exhibits non-monotonic convergence curve. Optimum at n_nodes≈185; coarser or finer grids degrade. Likely due to Jacobian ill-conditioning or incomplete Newton iteration budget as DOF increase.
- **Lesson**: Mesh refinement alone within scipy framework is futile. The 1.75e-12 plateau is a **solver algorithm limit**, not discretization. Confirmed agent5's Fourier breakthrough (exp085: 2.67e-13 via spectral method) as necessary approach.
- **Implication**: Config-only tuning (u_offset, n_nodes, solver_tol) is exhausted. Further progress requires algorithm change (Fourier, Newton-Krylov, preconditioner design).

## Excessive spectral resolution degrades Newton (agent5, exp098 confirms agent4's exp089)
- **Hypothesis**: fourier_modes=128 extends spectral accuracy beyond fourier_modes=64
- **Expected**: Exponential convergence continues; more modes = better residual (Boyd, Trefethen)
- **Actual**: exp098 (modes=128) → **CRASH**, Newton diverged after 200 iterations with residual=3.77e-12
- **Compared to**: exp087/exp097 (modes=32) → 2.36e-13 (15× better!)
- **Root cause**: Jacobian condition number κ grows with spectral basis size N. At modes=128 (256 DOF in physical space), Newton's system becomes ill-conditioned. Dealiasing rule (M=2N) introduces aliasing/coupling at high modes. Newton cannot make progress.
- **Lesson**: Spectral exponential convergence is achievable *within a regime* where Newton is stable. Beyond that regime (modes≥64 for this problem), condition number dominates and method fails.
- **Validated**: Confirmed agent4's finding — fourier_modes=32 is optimal. Do NOT use modes>64.
