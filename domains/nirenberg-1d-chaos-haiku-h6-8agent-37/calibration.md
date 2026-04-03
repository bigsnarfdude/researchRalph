# Calibration — nirenberg-1d-chaos-r3

## Benchmark identity

**Custom internal benchmark** — not a standard ML/NLP leaderboard. This is a nonlinear BVP solver optimization task:

- **Equation:** `u''(θ) = u³ - (1 + K(θ))·u`, 2π-periodic on S¹
- **K(θ)** = 0.3·cos(θ) (fixed)
- **Goal:** Find all 3 solution branches (trivial u≈0, positive u≈+1, negative u≈-1) with minimal RMS residual
- **Score:** Max RMS BVP residual (lower is better, 0 = exact)
- **Agents control:** Initial condition parameters (u_offset, amplitude, n_mode, phase) + solver settings
- **Two solver backends:** `scipy` solve_bvp (4th-order collocation) and `fourier` (pseudo-spectral Newton)

This is inspired by the Nirenberg prescribed curvature problem but reduced to 1D. The double-well structure `u³ - u` creates a potential with minima at u=±1 and saddle at u=0.

## Current SOTA (from prior runs)

### nirenberg-1d (control run, 700+ experiments)
| Branch | Best residual | Method | Config |
|--------|---------------|--------|--------|
| Trivial (u≈0) | **0.0** (exact zero) | scipy, n=196, tol=1e-12 | u_offset=0, amp=0 |
| Negative (u≈-1) | **9.17e-27** | scipy, n=300, tol=1e-11 | u_offset=-0.9, amp=0 |
| Positive (u≈+1) | **2.83e-22** | scipy, n=300, tol=1e-11 | u_offset=+0.9, amp=0 |

### nirenberg-1d-chaos (72 experiments, Fourier-focused)
| Branch | Best residual | Method | Config |
|--------|---------------|--------|--------|
| Trivial (u≈0) | **7.65e-23** | fourier, 64 modes, newton_tol=1e-12 | u_offset=0.46 (!) |
| Positive (u≈+1) | **5.55e-17** | fourier, **1 mode** | u_offset=+0.9 |
| Negative (u≈-1) | **5.55e-17** | fourier, **1 mode** | u_offset=-0.9 |

**Key observation:** Scipy dramatically outperforms Fourier spectral on non-trivial branches (1e-22 vs 1e-17). However, Fourier spectral found that **fewer modes = better** for non-trivial (1 mode: 5.55e-17 vs 64 modes: 3.5e-13). This counter-intuitive result suggests the Fourier Newton solver has conditioning issues at higher mode counts on non-trivial branches.

## Best known techniques

### What works (proven in prior runs)

1. **Scipy solver for non-trivial branches:** n=300, tol=1e-11, amp=0 → 1e-22 to 1e-27 residuals
2. **Scipy n_nodes=196 sweet spot:** For trivial branch, n=196 at tol=1e-12 gives exact zero. Monotone improvement 190→196, then cliff at 197.
3. **Fourier spectral for trivial branch:** 64 modes, newton_tol=1e-12, maxiter=100 → 1e-21 to 1e-23
4. **Fewer Fourier modes for non-trivial:** Counter-intuitively, 1-2 modes outperform 64 modes on ±1 branches
5. **amp=0 (flat IC):** Once in the right basin, zero oscillation amplitude works best
6. **Phase optimization:** phase=π for positive branch, phase=0 for negative (only matters at suboptimal configs)

### Technique-specific details

- **Basin boundaries are fractal:** u_offset 0.52–0.58 shows interleaved trivial/negative/positive basins. The Newton convergence basins are not simply connected. Fine boundary sweeps (0.01 steps) needed.
- **tol=1e-12 crashes non-trivial scipy:** The solver lacks sufficient DOF at tol=1e-12 for non-trivial. tol=1e-11 is the practical scipy limit.
- **Fourier newton_tol=1e-14 crashes:** Gets stuck at ~5e-13 with maxiter=50. Use newton_tol=1e-12 + maxiter=100.
- **Fourier modes > 64 crash on non-trivial:** 96 and 128 modes consistently crash. Likely conditioning issues with the dense Jacobian.

### Approaches from the literature

1. **Deflated Newton method** (Farrell et al., SIAM J. Sci. Comput. 2015): Systematically eliminates known solutions to find new ones. Could discover exotic branches if they exist. Not implemented in current solve.py.
2. **Spectral trust-region deflation** (J. Sci. Comput. 2023): Combines spectral methods with trust-region globalization for finding multiple solutions.
3. **Numerical continuation / pseudo-arclength:** Parameter continuation in K_amplitude could trace bifurcation diagram and discover branch connections.
4. **Dealiasing for cubic nonlinearity:** The current solver uses M=2N physical points. For cubic terms, full dealiasing requires (n+1)/2 = 2x factor, so M=2N is correct. However, the dense Jacobian construction (O(M²) memory, O(M³) solve) limits practical mode count.

## What has been tried and failed

### Definitively failed approaches
1. **Fourier modes > 64:** Modes 96, 128 all crash on non-trivial branches (conditioning collapse in dense Jacobian solve)
2. **newton_tol < 1e-12:** tol=1e-13 and 1e-14 fail to converge within reasonable iterations
3. **scipy tol=1e-12 on non-trivial:** Crashes — solver cannot satisfy this tolerance
4. **Searching for 4th+ branches:** At K_amplitude=0.3, modes 2, 3, extreme offsets (±1.5) all converge to the same 3 branches. No exotic branches exist at this parameter regime.
5. **n_nodes > 300 for scipy:** n=500 is WORSE than n=300 for non-trivial (5.6e-12 vs 3.25e-12) — conditioning issues
6. **Fine boundary sweeps for better residual:** Basin boundaries are fractal/chaotic but don't lead to lower-residual solutions — they just determine which branch you land on

### Partially explored (worth revisiting)
1. **Fourier modes 1-8 on non-trivial:** 1 mode gave 5.55e-17 — is there an optimal small mode count?
2. **Combined method:** Scipy warm-start → Fourier polish (or vice versa)
3. **Adaptive Newton damping / line search:** Current solver uses full Newton steps; damped Newton might help convergence at tighter tolerances

## Recommended starting point for this run

### Phase 1: Reproduce known results (3-5 experiments)
1. Trivial branch: scipy, n=196, tol=1e-12, u_offset=0, amp=0 → expect residual=0.0
2. Positive branch: scipy, n=300, tol=1e-11, u_offset=0.9, amp=0 → expect ~1e-22
3. Negative branch: scipy, n=300, tol=1e-11, u_offset=-0.9, amp=0 → expect ~1e-27
4. Positive branch: fourier, fourier_modes=1, newton_tol=1e-12, maxiter=100, u_offset=0.9 → expect ~5.55e-17

### Phase 2: Push residual floors (10-15 experiments)
Focus on closing the gap between trivial (exact 0.0) and non-trivial (1e-22):
- **Scipy param sweep:** n_nodes in [194, 195, 196, 197, 198, 200, 250, 300] at tol=1e-11 for non-trivial
- **Ultra-low Fourier modes:** fourier_modes in [1, 2, 3, 4] on non-trivial branches
- **Fourier trivial optimization:** modes in [8, 16, 32, 64], newton_tol=1e-14 with maxiter=200

### Phase 3: Novel approaches (15+ experiments)
- **Two-stage solve:** Scipy first (coarse), then Fourier polish on the converged solution
- **n_nodes fine grid near 196:** Test 196 ± variations with tol in [8e-12, 1e-11] band
- **Explore solver_tol in [1e-11, 5e-11] with different n_nodes:** The 10x gap between requested/achieved tol suggests there may be a different sweet spot

### What NOT to try
- Fourier modes > 64 (will crash)
- newton_tol < 1e-12 (will not converge)
- scipy tol < 1e-11 on non-trivial (will crash)
- Exotic branch searches at K_amplitude=0.3 (confirmed: only 3 branches exist)
- n_nodes > 300 for scipy (conditioning degrades)

## Sources searched

### Web searches performed
- [The Nirenberg problem and its generalizations — Math. Annalen](https://link.springer.com/article/10.1007/s00208-016-1477-z)
- [Continuation and Bifurcation in Nonlinear PDEs — Springer](https://link.springer.com/article/10.1365/s13291-021-00241-5)
- [Deflation techniques for finding distinct solutions — Farrell et al., SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/140984798)
- [Efficient Spectral Trust-Region Deflation Method — J. Sci. Comput. 2023](https://link.springer.com/article/10.1007/s10915-023-02154-0)
- [Two-Level Spectral Methods for Nonlinear Elliptic Equations with Multiple Solutions — SIAM](https://epubs.siam.org/doi/10.1137/17M113767X)
- [Global behavior of bifurcation curves for periodic nonlinear terms — CPAA](https://www.aimsciences.org/article/doi/10.3934/cpaa.2018102)
- [scipy.integrate.solve_bvp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)
- [De-aliasing in FFT — Fei Lu, JHU](https://math.jhu.edu/~feilu/notes/DealiasingFFT.pdf)
- [Boyd: Chebyshev and Fourier Spectral Methods](https://depts.washington.edu/~ph506/Boyd.pdf)
- [Chebfun guide: Nonlinear ODEs and BVPs](https://www.chebfun.org/docs/guide/guide10.html)
- [Computing nearly singular solutions using pseudo-spectral methods](https://arxiv.org/pdf/math/0701337)

### Internal data analyzed
- `nirenberg-1d/results.tsv` (700+ experiments, best: 0.0 trivial, 9.17e-27 non-trivial)
- `nirenberg-1d-chaos/results.tsv` (72 experiments, best: 5.55e-17 non-trivial via Fourier)
- `nirenberg-1d/LEARNINGS.md` (11 learnings across 3 generations)
- `nirenberg-1d/MISTAKES.md` (3 documented mistakes)
- `nirenberg-1d-chaos/LEARNINGS.md` + `MISTAKES.md`
- `nirenberg-1d-chaos-r3/solve.py` (Fourier spectral + scipy dual backend)
