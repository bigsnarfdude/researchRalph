# Calibration — nirenberg-1d-chaos-r4

Generated: 2026-04-02

## Benchmark identity

**Problem:** Double-well BVP on S^1 (periodic circle), inspired by Nirenberg curvature prescription:

    u''(theta) = u^3 - (1 + K(theta))*u,   theta in [0, 2*pi],   periodic BCs
    K(theta) = 0.3 * cos(theta)

Three known solution branches: trivial (u~0), positive (u~+1), negative (u~-1). Goal: find all three branches with minimal BVP residual. This is NOT a standardized benchmark — it's a custom domain for testing agent exploration of solution landscapes.

**Closest mathematical relatives:**
- Allen-Cahn / Ginzburg-Landau equation with double-well potential (u^3 - u term)
- Nirenberg problem of prescribed Gaussian curvature on S^2 (reduces to similar semilinear elliptic PDE)
- General semilinear elliptic BVPs with multiple solutions

## Current SOTA (from prior runs)

### Run r3 results (190 experiments, 4 agents):
| Branch | Best residual | Method | Experiment |
|--------|--------------|--------|------------|
| Trivial (mean~0) | **0.0** (exact) | scipy, n=196, tol=1e-12, amp=0 | exp001 |
| Positive (mean~+1) | **5.55e-17** (machine eps) | Fourier 1-mode, u_offset=0.9 | exp008 |
| Negative (mean~-1) | **5.55e-17** (machine eps) | Fourier 1-mode, u_offset=-0.9 | exp016 |

**All three branches have been solved to machine precision.** The residual floor is:
- Trivial: 0.0 (exact zero is a solution of the BVP)
- Non-trivial: 5.55e-17 = eps/2 for float64 (fundamental numerical floor for O(1) solutions)

### Key quantitative findings from r3:
- Fourier spectral solver beats scipy by 5 orders of magnitude on non-trivial branches (5.55e-17 vs 1.47e-12)
- Scipy best on non-trivial: 1.47e-12 (n=196, tol=1e-11)
- Fourier result is invariant to IC parameters within basin of attraction
- Fewer Fourier modes = better: 1 mode → 5.55e-17, 2 modes → 2.0e-16, 3 modes → 4.4e-16
- n_nodes=196 is a scipy sweet spot (non-monotonic: n=197 jumps to 1e-11)

## Best known techniques

### 1. Fourier pseudo-spectral + Newton (BEST — already in solve.py)
- Exponential (spectral) convergence for smooth periodic solutions
- Newton iteration with full Jacobian (circulant D^2 + diagonal)
- For this problem: converges in ~5-10 iterations to machine eps
- Key: use `method: fourier`, `fourier_modes: 64`, `newton_tol: 1e-14`
- **1 Fourier mode in the IC is optimal** — solution is nearly constant + small cos perturbation

### 2. Scipy solve_bvp (BACKUP — 4th order collocation)
- Algebraic convergence only → floor ~1e-12 on non-trivial
- Sweet spots: n_nodes=196, tol in [8e-12, 1e-11]
- Crashes at tol <= 3e-12 on non-trivial branches

### 3. Deflated Newton (ADVANCED — not yet implemented)
- Farrell, Birkisson & Funke (2015), SIAM J. Sci. Comput. 37:A2026-A2045
- Systematically eliminates known solutions to find additional ones from same IC
- Could discover exotic/unstable branches beyond the three known ones
- Recent extension: "Spectral Trust-Region Deflation" (2023) combines spectral Galerkin + deflation + trust-region

### 4. Branch switching at bifurcation points
- Liapunov-Schmidt reduction for symmetry-breaking bifurcations
- Continuation methods (pseudo-arclength) to trace solution curves as K_amplitude varies

### 5. PINN approach
- "A Machine Learning Approach to the Nirenberg Problem" (arxiv 2602.12368, Feb 2026)
- Mesh-free PINN achieving losses 1e-7 to 1e-10 on S^2 Nirenberg
- Not competitive with spectral methods for this 1D case

## What has been tried and failed

### From r3 MISTAKES.md:
1. **scipy tol=1e-12 on non-trivial branches → CRASH.** Boundary is between 3e-12 and 4e-12. Must use tol >= 4e-12 (optimal: 8e-12 to 1e-11).
2. **More Fourier modes in IC → worse residual.** The non-trivial solutions are nearly constant, so adding modes just adds noise.
3. **Varying IC parameters within basin → no improvement.** Fourier solver converges to same discrete solution regardless of starting u_offset, amplitude, phase within basin.

### Known dead ends for this run:
- **Tuning scipy parameters further** — ceiling is 1e-12, already found
- **Varying fourier_modes in IC beyond 1** — monotonically worse
- **Hunting for 4th+ branches with standard initial guesses** — the Z2 symmetry of the double-well guarantees exactly three branches for small K_amplitude. Higher bifurcations only appear at larger K_amplitude.
- **Tighter newton_tol** — already at machine eps floor
- **PINN/ML approaches** — orders of magnitude worse than spectral for smooth 1D periodic

## Recommended starting point for this run

**The residual minimization problem is SOLVED.** All three branches are at machine precision. The interesting research directions for r4 are:

### Option A: Solution space mapping (recommended)
- Systematically map the basins of attraction: for what range of u_offset does Newton converge to each branch?
- Find the basin boundaries precisely (bifurcation-like behavior in IC space)
- Vary K_amplitude and K_frequency to find new bifurcation points where additional branches appear

### Option B: Higher bifurcations
- Increase K_amplitude beyond 0.3 — at some critical value, mode-2 solutions should bifurcate off the ±1 branches
- This requires modifying K parameters (currently read-only in the config)

### Option C: Robustness testing
- Stress-test the Fourier solver: extreme n_mode (4, 8, 16), large amplitude, pathological phases
- Find initial conditions that cause Newton to fail or converge slowly
- Map the convergence rate (iterations to converge) as a function of IC parameters

### Recommended program.md guidance:
```
Start with Fourier method (method: fourier). All three branches are already solved to machine eps.
Focus on: (1) basin mapping — sweep u_offset finely to find branch boundaries,
(2) perturbation sensitivity — how much IC noise can the solver tolerate,
(3) if config allows: vary K_amplitude to find bifurcation thresholds.
Avoid: scipy parameter tuning (dead end), more Fourier modes in IC (worse).
```

## Sources searched

- [A Machine Learning Approach to the Nirenberg Problem (arxiv 2602.12368)](https://arxiv.org/abs/2602.12368) — PINN for Nirenberg on S^2, Feb 2026
- [Deflation techniques for distinct solutions of nonlinear PDEs (Farrell et al. 2015)](https://arxiv.org/abs/1410.5620) — foundational deflated Newton method
- [An Efficient Spectral Trust-Region Deflation Method (2023)](https://link.springer.com/article/10.1007/s10915-023-02154-0) — spectral Galerkin + deflation + trust-region
- [A spectral Levenberg-Marquardt-Deflation method (2025)](https://arxiv.org/html/2503.01912) — LM-deflation for semilinear elliptic systems
- [Structure probing neural network deflation (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0021999121001261) — NN deflation for multiple PDE solutions
- [scipy solve_bvp convergence issue #9832](https://github.com/scipy/scipy/issues/9832) — known BC convergence checking bug
- [Computation and stability of periodic orbits via Fourier/Chebyshev spectral (2024)](https://arxiv.org/html/2407.18230)
- [Anderson: The Nirenberg problem of prescribed Gauss curvature on S^2](https://www.math.stonybrook.edu/~anderson/nirenbfinal.pdf)
- [Allen-Cahn equation (Wikipedia)](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation) — double-well potential context
- [Boyd: Chebyshev and Fourier Spectral Methods](https://depts.washington.edu/ph506/Boyd.pdf) — reference on spectral accuracy
- Prior run data: nirenberg-1d-chaos-r3 (190 experiments, LEARNINGS.md, MISTAKES.md, results.tsv)
