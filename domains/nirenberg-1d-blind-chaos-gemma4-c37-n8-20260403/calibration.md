# Nirenberg 1D — Calibration

Generated: 2026-04-01

## Benchmark Identity

**Not a standard ML benchmark.** This is a computational mathematics problem: finding all solution branches of a 1D double-well BVP on the circle S^1.

**Equation:** `u''(θ) = u³ - (1 + K(θ))·u`, periodic on [0, 2π], with K(θ) = 0.3·cos(θ).

This is closely related to the **Allen-Cahn equation** (reaction-diffusion, Ginzburg-Landau free energy) and **Nirenberg's prescribed curvature problem** on S². The 1D reduction with periodic BCs and a double-well potential (u³ - u forcing) is a textbook nonlinear BVP with known multiplicity.

**Three solution branches exist** (for K_amplitude=0.3, small perturbation from K=0 case):
| Branch | solution_mean | Stability |
|--------|---------------|-----------|
| Trivial | ≈ 0.0 | Unstable saddle |
| Positive | ≈ +1.0 | Stable (minimum of double-well) |
| Negative | ≈ -1.0 | Stable (minimum of double-well) |

The ±1 branches are perturbations of the constant solutions u ≡ ±1 (minima of the double-well potential V(u) = u⁴/4 - u²/2). The trivial branch u ≡ 0 is the unstable equilibrium at the local maximum.

## Current SOTA (for this solver setup)

This is not a competitive benchmark — "SOTA" means "what residual is achievable."

- **Residual floor:** scipy's `solve_bvp` with tol=1e-8 achieves residuals of **~5e-11** on the trivial branch (exp001). This is essentially machine-precision-limited for float64 collocation.
- **Positive branch:** exp002 achieved **5.7e-9** — slightly higher residual due to the nontrivial solution shape requiring more collocation nodes.
- **Negative branch:** Not yet found in current run.
- **Theoretical limit:** For a 4th-order collocation method (Lobatto IIIa) with 100 nodes on a smooth periodic solution, residuals ~1e-10 to 1e-12 are achievable. Going below requires higher n_nodes or tighter solver_tol.

**Reference numbers from existing experiments:**
| exp_id | residual | mean | branch |
|--------|----------|------|--------|
| exp001 | 5.64e-11 | -0.0 | trivial |
| exp002 | 5.73e-09 | +1.0 | positive |

## Best Known Techniques

### Branch-finding (the primary goal)

1. **u_offset sweep:** The key branch selector. u_offset near 0 → trivial, near +0.9 → positive, near -0.9 → negative. Systematic sweep from -1.5 to +1.5 maps all branches.

2. **Initial guess quality matters:** scipy's solve_bvp uses a **damped Newton method** on the collocation system. Convergence basin depends on how close the initial guess is to the true solution. For the ±1 branches, u_offset ≈ ±0.9 is the sweet spot (close to the true constant solution ±1, perturbed by K).

3. **Fourier mode perturbations:** Since K(θ) = 0.3·cos(θ) has mode-1, the solution perturbation from the constant is also primarily mode-1. Setting n_mode=1 with small amplitude (0.05–0.15) and matching phase should help convergence. For mode-2 or mode-3 initial conditions, the solver may find the same branches but convergence could be slower or fail.

4. **Bifurcation boundary detection:** The transition between branches occurs at some critical |u_offset|. Binary search in the range 0.3–0.7 can pinpoint where the solver switches from trivial to nontrivial branch.

### Residual minimization (secondary goal)

5. **Increase n_nodes:** More collocation points → lower residual, at marginal cost (still sub-second). Try 150, 200, 300 nodes.

6. **Tighten solver_tol:** Going from 1e-8 to 1e-10 may improve residual by 1-2 orders of magnitude, if the solver converges.

7. **Smart initial guess:** Setting amplitude to match the expected perturbation shape (K_amplitude-dependent) gives the solver a head start. For K_amplitude=0.3, the perturbation from u=±1 is O(0.3), so amplitude ≈ 0.1–0.15 with phase matching K's cosine is optimal.

### Advanced (if agents exhaust basic sweeps)

8. **Numerical continuation:** Vary K_amplitude from 0 to 0.3 in steps, using each converged solution as the initial guess for the next. This is what AUTO-07p and pde2path do. Not available in this harness, but agents could approximate it by running multiple experiments with different K settings if allowed.

9. **Energy landscape analysis:** The solution_energy output distinguishes branches even when residuals are similar. The positive and negative branches have energy ≈ -1.52 (from exp002), while trivial has energy ≈ 0.

## What Has Been Tried and Failed (Known Failure Modes)

### For this type of problem generally:
- **Zero initial guess for nontrivial branches:** u_offset=0 with zero amplitude always converges to the trivial branch. The Newton iteration has no reason to leave the basin of u≡0.
- **Large amplitude perturbations (>0.5):** Can push the initial guess outside all convergence basins, causing solve_bvp to fail or return garbage.
- **Wrong Fourier mode:** n_mode=2 or 3 initial guesses add spatial oscillation that the ±1 branches don't have. May still converge but wastes Newton iterations.
- **Too tight tolerance without enough nodes:** solver_tol=1e-12 with n_nodes=50 may fail to converge — not enough degrees of freedom to represent the solution to that accuracy.
- **Expecting exotic branches:** For K_amplitude=0.3 (small), there are exactly three branches. Higher K_amplitude could create additional bifurcated branches, but K is fixed in this problem. Mode-2 initial conditions won't find a "mode-2 branch" — there isn't one for these parameters.

### From current run:
- Trivial branch (exp001) and positive branch (exp002) already found.
- Negative branch is the immediate gap.

## Recommended Starting Point for This Run

### Phase 1: Complete branch coverage (immediate priority)
1. **Find negative branch:** u_offset = -0.9, amplitude = 0.1, n_mode = 1, phase = 0.0 → expect mean ≈ -1.0
2. **Verify symmetry:** Compare positive and negative branch residuals and energies. By the Z₂ symmetry u → -u of the equation (when K has the right parity), the branches should be mirror images.

### Phase 2: Branch boundary mapping
3. **Sweep u_offset** from 0.0 to +1.0 in steps of 0.1 to find the transition point where solver switches from trivial to positive branch.
4. **Binary search** the transition boundary to 2 decimal places.
5. **Repeat for negative side** (u_offset 0.0 to -1.0).

### Phase 3: Residual optimization
6. **Increase n_nodes** to 200 on each branch to push residuals lower.
7. **Tighten solver_tol** to 1e-10 with n_nodes=200.
8. **Optimize initial guess phase:** Since K(θ) = 0.3·cos(θ), try phase = 0 and phase = π to see which gives better convergence on each branch.

### Phase 4: Exotic exploration (if time permits)
9. **n_mode=2 and n_mode=3** initial conditions — confirm they converge to the same three branches (no hidden mode-2 solutions for these parameters).
10. **Extreme u_offset** values (±1.5) — check convergence behavior at parameter boundaries.

### Expected timeline:
Each experiment takes <1 second. Two agents can cover all three phases in ~30-50 experiments total. This is a fast domain — the bottleneck is agent thinking time, not compute.

## Sources Searched

- [Bifurcation in Nirenberg's problem — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0764444298850118)
- [Bifurcation of radial solutions for prescribed mean curvature equations — arXiv 2603.12952](https://arxiv.org/abs/2603.12952)
- [The Nirenberg Problem of Prescribed Gauss Curvature on S² — Anderson](https://www.math.stonybrook.edu/~anderson/nirenbfinal.pdf)
- [scipy.integrate.solve_bvp — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)
- [An Allen-Cahn equation with continuation — Chebfun](https://www.chebfun.org/examples/ode-nonlin/AllenCahn.html)
- [Periodic solutions for the Allen-Cahn equation — Advances in Continuous and Discrete Models](https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-015-0631-3)
- [pde2path — Matlab package for continuation and bifurcation in PDEs](https://pde2path.uol.de/)
- [Numerical Continuation and Bifurcation in Nonlinear PDEs — SIAM](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976618.fm)
- [auto-AUTO: Python layer for AUTO-07p — JOSS 2025](https://joss.theoj.org/papers/10.21105/joss.08079.pdf)
- [Solve BVP with Multiple Boundary Conditions — MATLAB bvp4c](https://www.mathworks.com/help/matlab/math/solve-bvp-with-multiple-boundary-conditions.html)
- [BVP examples with solve_bvp — Kitchin Group, CMU](https://kitchingroup.cheme.cmu.edu/pycse/book/09-bvp.html)
- [Numerical Continuation and SPDE Stability for 2D Allen-Cahn — SIAM/ASA JUQ](https://epubs.siam.org/doi/10.1137/140993685)
