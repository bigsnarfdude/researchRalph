# Calibration — nirenberg-1d-chaos

Generated: 2026-04-02

## Benchmark identity

This is **not** a standard ML benchmark. It is a custom numerical analysis domain: finding all solution branches of a nonlinear periodic BVP on S¹ (the circle):

    u''(θ) = u³ - (1 + K(θ))·u,   θ ∈ [0, 2π],   periodic BCs
    K(θ) = 0.3·cos(θ)

This is a **double-well / Duffing-type** equation with spatially varying coefficient. The double-well potential V(u) = u⁴/4 - u²/2 has minima at u = ±1, so the equation admits three solution branches:
- **Trivial**: u ≡ 0 (unstable equilibrium of the well)
- **Positive**: u ≈ +1 (near the positive well minimum)
- **Negative**: u ≈ -1 (near the negative well minimum)

The problem is inspired by the **Nirenberg curvature prescription problem** on S² (prescribing Gaussian curvature for conformal metrics), reduced to a 1D analogue on S¹.

**Score**: RMS BVP residual (lower is better, 0 = exact). Goal is not just low residual but **mapping all three branches**.

## Current SOTA (with numbers and citations)

There is no public leaderboard for this specific 1D problem. However, relevant baselines:

| Method | Expected residual | Notes |
|--------|-------------------|-------|
| Fourier pseudo-spectral + Newton (64 modes) | ~1e-14 | Machine precision for smooth periodic problems. Spectral methods converge exponentially. |
| scipy `solve_bvp` (collocation, 100 nodes) | ~1e-6 to 1e-8 | 4th-order algebraic convergence. Adequate but far from spectral. |
| PINN (Nirenberg Neural Network, arxiv:2602.12368) | ~1e-7 to 1e-10 | For the 2D Nirenberg problem on S². Mesh-free but slower convergence than classical spectral. |

**Key reference**: "A Machine Learning Approach to the Nirenberg Problem" (arxiv:2602.12368, Feb 2026) — first PINN approach to Nirenberg problem, achieves 1e-7 to 1e-10 loss for realisable curvatures on S².

For the **1D periodic Duffing-type BVP**, classical Fourier-Galerkin + Newton is the gold standard. With 64+ Fourier modes and Newton converging to tolerance 1e-14, residuals should be at or near machine epsilon (~1e-15 to 1e-14).

## Best known techniques (specific tactics, strategies, approaches)

### For finding all three branches

1. **Initial guess is everything**: The basin of attraction for each branch is determined almost entirely by `u_offset`:
   - u_offset ≈ 0.0 → trivial branch (u ≡ 0)
   - u_offset ≈ +0.9 → positive branch (u ≈ +1)
   - u_offset ≈ -0.9 → negative branch (u ≈ -1)

2. **Deflation technique** (Farrell et al., SIAM J. Sci. Comput. 2015): After finding one solution u*, modify the residual F(u) → F(u)/||u - u*||^p to deflate away known solutions. This lets Newton converge to new branches from the **same initial guess**. Most principled approach for systematically finding all branches.

3. **Continuation / homotopy**: Parameterize K_amplitude from 0 (where branches are obvious: u = 0, ±1 exactly) up to 0.3. Track branches via pseudo-arclength continuation. Overkill for this problem since good initial guesses suffice.

4. **Symmetry exploitation**: The equation has u → -u symmetry when K(θ) = K(-θ) (which holds for K = 0.3·cos(θ)). So the negative branch is the negative of the positive branch. Only need to find 2 distinct solutions + their negative.

### For minimizing residual

1. **Fourier pseudo-spectral method** (already in solve.py): Exponential convergence for smooth periodic problems. The solver already implements this.
   - Use sufficient modes: 64 modes is likely more than enough for K_amplitude = 0.3
   - Newton tolerance 1e-14 should yield residuals near machine epsilon

2. **Dealiasing**: The 3/2 rule (or better, Fourier smoothing per Hou & Li 2007) for the u³ nonlinearity. The current solver uses 2N physical points for N modes, which is adequate.

3. **scipy solve_bvp** (also in solve.py): Algebraic convergence only. Max residual ~1e-6 to 1e-8. Useful as cross-check but not for lowest residuals.

4. **Key Newton convergence tips**:
   - If Newton diverges, the initial guess is too far from a solution — adjust u_offset
   - Jacobian is well-conditioned near the three equilibria (K_amplitude = 0.3 is mild)
   - Damped Newton (line search) can help if starting far from solution, but shouldn't be needed here

### Solver parameters that matter

| Parameter | Good default | Notes |
|-----------|-------------|-------|
| `u_offset` | 0.0, +0.9, -0.9 | Branch selector. Most important parameter. |
| `amplitude` | 0.01–0.1 | Small perturbation of initial guess. Too large → wrong basin. |
| `n_mode` | 1 | Match K_frequency for best initial guess shape |
| `fourier_modes` | 64 | Overkill for this smooth problem (32 probably suffices) |
| `newton_tol` | 1e-14 | Machine precision target |
| `n_nodes` | 100–200 | Only matters for scipy solver |

## What has been tried and failed

### Known failure modes for this class of problem

1. **Wrong initial guess → wrong branch**: Starting with u_offset = 0.5 may converge to trivial OR positive branch unpredictably. The basins of attraction have fractal boundaries. Stay near 0.0 or ±0.9.

2. **Too few Fourier modes → aliasing artifacts**: With < 16 modes, the u³ term aliases and Newton may diverge or converge to spurious solutions.

3. **Newton divergence from flat initial guess**: Starting with u_offset = 0.0 and amplitude = 0.0 gives u ≡ 0, which is already a solution (trivial branch). Newton will just stay there. Need nonzero amplitude to escape.

4. **scipy solve_bvp with periodic BCs**: scipy's `solve_bvp` doesn't natively support periodic BCs — they must be encoded as bc(ya, yb) = [ya[0]-yb[0], ya[1]-yb[1]]. This works but is less natural than Fourier methods. Can also fail to converge if initial mesh is too coarse.

5. **Trying to optimize the K function**: K_mode, K_amplitude, K_frequency are READ-ONLY. Changing them changes the problem itself, invalidating all comparisons.

6. **Over-engineering the initial guess**: The problem is well-conditioned for K_amplitude = 0.3. Sophisticated continuation or deflation is unnecessary — simple offset-based targeting works reliably.

7. **Confusing "lower residual" with "better exploration"**: Finding u ≡ 0 with residual 1e-15 is not better than finding all three branches with residual 1e-10 each. The research goal is branch coverage + low residual.

## Recommended starting point for this run

### Phase 1: Branch discovery (experiments 1-6)
1. **Trivial branch**: u_offset=0.0, amplitude=0.1, method=fourier → expect residual ~1e-14, mean≈0
2. **Positive branch**: u_offset=+0.9, amplitude=0.1 → expect residual ~1e-14, mean≈+1
3. **Negative branch**: u_offset=-0.9, amplitude=0.1 → expect residual ~1e-14, mean≈-1
4. Cross-check with scipy: same three offsets with method=scipy → expect residual ~1e-6 to 1e-8
5. Verify symmetry: positive and negative branch solutions should be negatives of each other

### Phase 2: Residual optimization (experiments 7-15)
6. Fourier mode sweep: 16, 32, 64, 128 modes to find diminishing returns
7. Newton tolerance sweep: 1e-10, 1e-12, 1e-14 to verify convergence
8. Initial guess refinement: try amplitude=0.01 vs 0.1 vs 0.3 — does it affect final residual?
9. n_mode=1 vs 2: does matching the K_frequency help convergence speed?

### Phase 3: Edge cases and exploration (experiments 16+)
10. Boundary of basins: sweep u_offset from 0.3 to 0.7 to find the bifurcation point
11. Higher Fourier modes in initial guess: n_mode=2,3 with various offsets
12. Phase sensitivity: does phase shift of initial guess matter?

### Expected best achievable scores
- **Fourier method**: residual ≈ 1e-14 to 1e-15 (machine precision) for all three branches
- **scipy method**: residual ≈ 1e-6 to 1e-8 depending on mesh density and tolerance
- The agents should quickly achieve near-optimal residuals; the real challenge is systematic branch coverage and understanding the solution landscape

## Sources searched

### Nirenberg problem and curvature prescription
- [A Machine Learning Approach to the Nirenberg Problem (arxiv:2602.12368)](https://arxiv.org/abs/2602.12368) — PINN approach, Feb 2026
- [The Nirenberg problem and its generalizations (Springer)](https://link.springer.com/article/10.1007/s00208-016-1477-z) — unified approach
- [The Nirenberg problem of prescribed Gauss curvature on S² (arxiv:1707.02938)](https://arxiv.org/abs/1707.02938)

### Spectral methods for periodic BVPs
- [Computation and stability analysis of periodic orbits using Fourier spectral expansions (arxiv:2407.18230)](https://arxiv.org/html/2407.18230)
- [Stability and spectral convergence of Fourier method for nonlinear problems (arxiv:1308.5314)](https://arxiv.org/abs/1308.5314) — 2/3 dealiasing limitations
- [Computing nearly singular solutions using pseudo-spectral methods (Hou & Li 2007)](https://www.math.umd.edu/~tadmor/references/files/Hou%20&%20Li%20filter%20vs%202-3%20rule.pdf) — Fourier smoothing > 2/3 rule
- [B-spline periodization of Fourier pseudo-spectral method (arxiv:2512.06631)](https://arxiv.org/html/2512.06631)
- [Chebyshev and Fourier Spectral Methods (Boyd)](https://depts.washington.edu/ph506/Boyd.pdf) — textbook reference

### Finding multiple solutions / deflation
- [Deflation techniques for finding distinct solutions of nonlinear PDEs (Farrell et al., SIAM 2015)](https://epubs.siam.org/doi/abs/10.1137/140984798) — key paper on deflation
- [Deflation techniques for multiple local minima (arxiv:2409.14438)](https://arxiv.org/html/2409.14438)
- [Bifurcation curve detection with deflation (arxiv:2602.12940)](https://arxiv.org/pdf/2602.12940)
- [Continuation and Bifurcation in Nonlinear PDEs (Springer 2021)](https://link.springer.com/article/10.1365/s13291-021-00241-5)

### Double-well / Duffing equation
- [Duffing equation (Wikipedia)](https://en.wikipedia.org/wiki/Duffing_equation)
- [Explicit and exact solutions to cubic Duffing and double-well Duffing equations (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0895717710004425)
- [Dynamics of the Double-Well Duffing System (ITM 2025)](https://www.itm-conferences.org/articles/itmconf/pdf/2025/06/itmconf_iconmaa25_02002.pdf)

### scipy solve_bvp
- [scipy.integrate.solve_bvp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)
- [solve_bvp convergence issue #9832](https://github.com/scipy/scipy/issues/9832)

### Neural operators / HuggingFace papers
- [Fourier Neural Operators spectral perspective (huggingface.co/papers/2404.07200)](https://huggingface.co/papers/2404.07200)
- Not directly applicable — this domain uses classical numerical methods, not learned solvers
