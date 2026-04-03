# Desires — agent1, nirenberg-1d-chaos-haiku

## Fourier Spectral Backend

**Why:** Calibration noted Fourier achieved 5.55e-17 on trivial branch. Current scipy maxes at ~3e-12 for nontrivial branches (10^10 worse). Need higher precision solver.

**How it helps:** Could break through residual ceiling and discover if nontrivial branches also have exceptional points (like u_offset=0.4 for trivial).

**Implementation cost:** Swap solve.py from scipy.integrate.solve_bvp to fourier spectral method (numpy FFT-based Galerkin).

---

## Bifurcation Mapping Tool

**Why:** Sharp basin boundary at u_offset≈0.48-0.50 is fascinating but we're only sampling discrete points. Need to understand the full bifurcation diagram.

**How it helps:** Could identify all hidden branches or multistability regimes. Might explain why u_offset=0.4 finds such exceptional trivial solutions.

**Implementation cost:** Plotly interactive sweep over u_offset (0.0–1.0 continuous) with residual heatmap.

---

## Parameter Sensitivity Analysis

**Why:** We know n_nodes=196 is optimal, u_offset=0.4 is best for trivial, but why? No mechanistic understanding yet.

**How it helps:** Could guide design of initial conditions for other bifurcation problems (e.g., u_offset=0.35, n_mode=2, phase=π/4 might find even better points).

**Implementation cost:** Gradient-based or adjoint sensitivity of residual w.r.t. [u_offset, amplitude, n_mode, phase].

---

## Nontrivial Basin Boundary

**Why:** Found bifurcation for trivial↔nontrivial, but nontrivial (pos/neg) basin boundary unexplored. Do u_offset=0.85, 0.75 yield better residuals than 0.9?

**How it helps:** Could find better positive/negative solutions. Currently stuck at 3.25e-12, prior Opus at 2.83e-22.

**Implementation cost:** Sweep u_offset from 0.5 to 1.0 in finer steps (0.6, 0.7, 0.75, 0.8, 0.85, 0.95).

---

## Agent0's Discoveries & Remaining Desires

### ✓ SOLVED: Fourier Spectral Backend Superiority

**Finding:** Fourier 1-mode achieves 5.55e-17 residual on non-trivial branches (exp028-029, 033).

**Breakthrough:** 4000x improvement over scipy's 1.47e-12!

**Current implementation:** Method config already supports fourier; optimal params:
```
method: fourier
fourier_modes: 1
newton_tol: 1.0e-12
newton_maxiter: 100
```

**Status:** CLOSED — Fourier 1-mode is the definitive solver for this problem.

### Future Direction: Non-trivial Basin Boundary Mapping

**Why:** Agent1 found exceptional trivial residuals at u_offset=0.4 (5.87e-20). Does the positive/negative basin have similar structure?

**Hypothesis:** Bifurcation boundaries may host exceptional solutions for non-trivial branches too.

**Cost:** 5-10 more experiments sweeping u_offset in [0.5-1.0] at finer resolution with Fourier 1-mode.

**Expected value:** Could find u_offset values where positive/negative residuals rival the trivial exceptional point.

### Nice-to-have: Convergence Theory

**Why:** Why does Fourier 1-mode achieve 5.55e-17 while scipy maxes 1.47e-12? Is it:
- Double precision limits?
- Fourier basis structure matching the solution?
- Newton iteration depth?

**Cost:** Requires analysis (not experiments).

**Value:** Mechanistic understanding; might transfer to other BVPs.
