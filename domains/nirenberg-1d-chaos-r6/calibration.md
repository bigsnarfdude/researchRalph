# Calibration — nirenberg-1d-chaos-r6

## Benchmark identity

**Problem:** 1D double-well BVP on S¹ (periodic circle), inspired by Nirenberg curvature prescription:

    u''(θ) = u³ - (1 + K(θ))·u,   θ ∈ [0, 2π],   periodic BCs
    K(θ) = 0.3·cos(θ)

This is structurally equivalent to a **stationary Allen-Cahn equation** with spatially-varying potential on the circle. Three solution branches exist: trivial (u≡0), positive (mean≈+1), negative (mean≈-1). The task is to map all branches with minimal residual.

**Not a standard benchmark** — this is a custom domain designed for the researchRalph multi-agent framework. No external leaderboard exists. "SOTA" is defined by prior runs in this repository.

## Current SOTA (internal, from prior runs)

| Branch | Best Residual | Method | Source |
|--------|--------------|--------|--------|
| Trivial (mean≈0) | **0.0 (exact)** | Fourier spectral, u_offset=0, amp=0 | nirenberg-1d exp220+ |
| Negative (mean≈-1) | **9.17e-27** | Fourier spectral, fourier_modes=1 | nirenberg-1d |
| Positive (mean≈+1) | **2.83e-22** | Fourier spectral | nirenberg-1d |

Across 700 experiments in nirenberg-1d, the Fourier spectral solver with Newton iteration dominates. The scipy solve_bvp solver tops out around 1e-12 residual (4th-order algebraic convergence vs spectral exponential convergence).

**Chaos experiment context:** This is run 6 of the chaos agent series (3/8 agents = 37.5% are chaos agents). Prior chaos runs:
- r3: 191 experiments
- r4: 250 experiments (evenness 1.000)
- r5: 80 experiments

## Best known techniques (from 700+ prior experiments)

### What works — Fourier spectral method (already in solve.py)
1. **`method: fourier`** in config.yaml — switches from scipy to Fourier pseudo-spectral Newton solver
2. **`fourier_modes: 1`** is sufficient — the equation has smooth periodic solutions, even 1 Fourier mode captures the structure. Higher modes (4, 8, 16, 64) also work but add no benefit
3. **`newton_tol: 1e-14` to `1e-15`** — tighter tolerance drives residuals to machine epsilon
4. **`newton_maxiter: 50-200`** — more iterations help for harder initial guesses

### Branch selection — the key parameter
- **`u_offset ≈ 0.0`** → trivial branch (always converges, residual = 0.0 exactly)
- **`u_offset ≈ +0.9`** → positive branch (mean≈+1.0)
- **`u_offset ≈ -0.9`** → negative branch (mean≈-1.0)
- **Basin boundary** at |u_offset| ≈ 0.47 — below this, Newton falls to trivial; above, to ±1 branch

### Secondary parameters (marginal impact)
- `amplitude`: 0.0–0.1 typical, perturbation of initial guess. amp=0 works fine for Fourier method
- `n_mode`: 1 is default, 2-3 explored but no benefit found
- `phase`: no significant effect on converged solution
- `n_nodes`: only matters for scipy solver (100–300)

## What has been tried and failed

1. **Searching for a 4th branch** — extensive exploration in nirenberg-1d (experiments 600-700) tried exotic initial conditions, high Fourier modes, mode-2/3 perturbations. Result: only 3 branches exist. No 4th branch was found. The norm≈0.07 "solutions" are just the trivial branch with small perturbation artifacts.

2. **scipy solver for high precision** — caps out at ~1e-12 residual. Not competitive with Fourier spectral.

3. **Extremely high fourier_modes** (64, 128) — no benefit over modes=1-8. The solutions are smooth and low-frequency.

4. **Large amplitudes** (>0.3) — can cause Newton convergence failure (crash). Keep amplitude ≤ 0.2.

5. **u_offset near basin boundary** (0.45-0.48) — unpredictable convergence, sometimes trivial, sometimes ±1. Not useful for reliable branch targeting.

6. **Deflation / branch switching** — not implemented in solve.py (agents cannot edit it). The existing solver finds whichever branch the initial guess is closest to.

## Recommended starting point for this run

Since this is a **chaos agent experiment** (testing whether 37.5% adversarial agents disrupt herd quality), the scientific value is in the multi-agent dynamics, not in pushing residuals lower. The problem is effectively solved:

1. **Agents should use `method: fourier`** with `fourier_modes: 1` or `fourier_modes: 4`
2. **Sweep u_offset**: 0.0 (trivial), +0.9 (positive), -0.9 (negative) to cover all branches
3. **Set `newton_tol: 1e-14`** for near-machine-epsilon residuals
4. **Expected best residuals**: 0.0 (trivial), ~1e-20 to 1e-27 (non-trivial)
5. **Branch coverage is the real metric** — agents that find all 3 branches are succeeding

The interesting signal from this run is whether chaos agents (37.5%) cause oracle agents to waste cycles, chase phantom branches, or degrade the collective blackboard quality.

## Sources searched

### Web searches performed
- Nirenberg curvature prescription equation BVP periodic solutions bifurcation 2024 2025
- Double-well BVP periodic boundary conditions solution branches Newton spectral method
- Fourier spectral method nonlinear BVP Newton iteration convergence tricks 2024 2025
- scipy solve_bvp periodic boundary conditions multiple solutions branch switching
- Nonlinear BVP curvature prescription S1 circle all solution branches numerical continuation deflation
- Deflated Newton method finding multiple solutions BVP nonlinear PDE 2024
- Chebfun bifurcation periodic BVP double-well multiple solutions numerical
- "u'' = u^3 - (1+K)u" periodic solution bifurcation numerical
- Allen-Cahn equation periodic boundary spectral method solution branches numerical 2024

### Relevant external references
- [Deflation techniques for finding distinct solutions of nonlinear PDEs (Farrell et al., SIAM J. Sci. Comput. 2015)](https://epubs.siam.org/doi/abs/10.1137/140984798)
- [Continuation and bifurcation in nonlinear PDEs (Uecker, 2021)](https://link.springer.com/article/10.1365/s13291-021-00241-5)
- [Accelerating Newton's method for nonlinear elliptic PDEs using FNO (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1007570424006191)
- [Fourier-spectral method for phase-field equations (MDPI, 2020)](https://www.mdpi.com/2227-7390/8/8/1385)
- [Solving Allen-Cahn equations with periodic BCs using mimetic FD (2024)](https://pubmed.ncbi.nlm.nih.gov/39669103/)
- [Chebfun nonlinear ODE guide — multiple BVP solutions](https://www.chebfun.org/docs/guide/guide10.html)
- [scipy.integrate.solve_bvp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)

### Internal data sources
- nirenberg-1d: 700 experiments (the primary reference run)
- nirenberg-1d-chaos: 73 experiments (first chaos test)
- nirenberg-1d-chaos-r3 through r5: 191, 250, 80 experiments
- nirenberg-1d-blind / nirenberg-1d-blind-chaos: 30, 27 experiments (no-oracle variants)
