# Meta-Blackboard — BVP Solver Benchmark Cheat Sheet

374 experiments. 2 agents. Key: lower residual = better.

## Winning recipe

**Fourier spectral, N=2 modes, positive branch** (exp373): residual **2.00e-16**

Not yet captured in best/config.yaml. The stored config (n_nodes=198, scipy, tol=1e-11) reflects the scipy-era best of ~1.47e-12, which is 4 OOM worse.

**Scipy best** (exp089/090): constant u=±1 solutions achieve 2.83e-22 / 9.17e-27 — but these are trivial exact solutions, not the oscillating branch the benchmark cares about.

Validate first: Fourier N=2, positive branch, newton_tol=1e-12, u_offset=0.9. Should reproduce 2.00e-16. [HIGH confidence]

## What works (ranked by impact)

| Rank | Technique | Gain | Why |
|------|-----------|------|-----|
| 1 | **Fourier spectral method** | 1.47e-12 → 2.88e-13 (5x at N=64), → 2.00e-16 (7000x at N=2) | Spectral convergence: smooth periodic solutions need few modes. Scipy's finite-difference discretization introduces O(h^p) error that floors the residual. |
| 2 | **Fewer Fourier modes** (N=2-4 beat N=48-128) | N=64: 2.88e-13 → N=2: 2.00e-16 (1400x) | Solution is essentially u ≈ 1 + ε·cos(t). Two Fourier coefficients capture it exactly. More modes add noise from Newton conditioning. |
| 3 | **Magic node count n=196** (scipy only) | 1.58e-12 (n=191) → 1.47e-12 (n=196) | Mesh-dependent aliasing. n=196 minimizes discretization artifacts for this K(t)=0.3cos(t) potential. Verified: n=392 (2×196) gives 1.46e-12, marginal gain. |
| 4 | **amp=0 constant initial guess** | Eliminates dependence on perturbation shape | For constant u=±1 solutions, exact IC converges in fewer Newton steps. For oscillating branch, amp=0.1 and amp=0 give identical residuals at optimal n. |
| 5 | **tol=1e-11 sweet spot** (scipy) | tol=1e-10: ~9e-11; tol=1e-11: ~1.47e-12 | Tighter than 4e-12 crashes (Newton diverges). 1e-11 is the tightest stable tolerance. |

## Dead ends

**Solver parameter grinding** (60+ experiments, no gain past 1.47e-12):
- n=500/1000/2000 scipy: crashes or worse (exp289: crash at n=2000; exp286: 5.59e-12 at n=500)
- tol < 4e-12 scipy: universal crash on nontrivial branch (exp244,259,260,264,301,302)
- tol=5e-12 to 7e-12: stuck at 3.47e-12 plateau (exp235,242,246,248)

**Exotic branch hunting** (30+ experiments, no new branches found):
- Mode-2/3 perturbations: all collapse to trivial u=0 (exp051,114,275,299,368)
- Heteroclinic/kink guesses: collapse to trivial or known branches (exp113,270,347,349)
- Large-amplitude oscillating ICs (amp=0.5-1.2): no new solutions (exp272,351)
- Extreme u_offset (±1.5): same constant branches (exp052-055,094-095)

**Boundary refinement** (80+ experiments, pure waste):
- Binary search of bifurcation boundary (u_offset=0.52-0.57): maps basin structure but finds no new solutions. Boundary is fractal-like with chaotic basin assignment.

**Fourier at high N**: N=96,128 crash (exp310,311,319,356,358). Conditioning degrades.

## Scaling laws

**Scipy: n_nodes vs residual** (tol=1e-11, positive branch)

| n_nodes | Residual | Notes |
|---------|----------|-------|
| 128 | 1.58e-12 | |
| 150 | 7.78e-12 | Non-monotone! |
| 195-196 | 1.47e-12 | **Optimum** |
| 200 | 1.00e-11 | Worse (mesh mismatch) |
| 300 | 3.25e-12 | |
| 392 | 1.46e-12 | 2×196, marginal gain |
| 500 | 5.59e-12 | |

Non-monotone relationship. n=196 is a resonance, not a smooth curve.

**Fourier: N (mode count) vs residual** (positive branch)

| N | Residual | Notes |
|---|----------|-------|
| 2 | 2.00e-16 | **Best** (slightly different solution_norm) |
| 3 | 4.38e-16 | |
| 4 | 2.58e-16 | |
| 8 | 2.95e-15 | |
| 16 | 5.53e-15 | |
| 32 | 5.49e-14 to 5.88e-13 | newton_tol-dependent |
| 48 | 1.35e-13 | Sweet spot for moderate N |
| 64 | 2.88e-13 | |
| 128 | crash | |

**Scipy tolerance thresholds** (n=196, positive branch)

| tol range | Residual | Behavior |
|-----------|----------|----------|
| ≤ 3e-12 | crash | Newton diverges |
| 4e-12–7e-12 | 3.47e-12 | Lower plateau |
| 8e-12–1e-11 | 1.47e-12 | **Upper plateau (best)** |
| 1.5e-11–2e-11 | 1.17e-11 | Degraded |
| 1e-10 | 9.37e-11 | Much worse |

## Stepping stones

- **exp300 (Fourier N=64, 2.88e-13)**: First proof that spectral methods break the scipy floor. Led to the N-sweep that found N=2 optimum.
- **exp089 (constant u=1, 2.83e-22)**: Proved the constant solution is exact to machine precision. Separated "solver accuracy" from "solution accuracy."
- **exp043 (amp=0.15, 9.99e-11, energy=-1.559)**: An intermediate solution between constant and oscillating branches. Energy -1.559 vs -1.524 (oscillating) vs -1.571 (constant). Never explored further.
- **exp373 (Fourier N=2, norm=1.001298 vs N≥3 norm=1.001296)**: Slightly different solution properties suggest N=2 may converge to a marginally different point on the solution manifold.

## Blind spots

1. **Fourier with custom Newton tolerance**: N=2 used default newton_tol. Sweeping newton_tol at N=2-4 could push below 2e-16. [HIGH promise]
2. **Chebyshev/collocation spectral methods**: Only Fourier tried. Chebyshev may handle boundary layers better if they exist. [MEDIUM promise]
3. **Continuation/homotopy methods**: Tracking branches as K_amplitude varies from 0→0.3 would systematically find all bifurcations. Never attempted. [HIGH promise for finding new branches]
4. **K_amplitude/K_frequency variation**: All 374 experiments used K=0.3cos(t). Different K parameters might reveal richer solution structure. [MEDIUM, if allowed by harness]
5. **Multi-precision arithmetic**: mpmath or similar to push past float64 limits. [LOW, likely overkill]
6. **Deflation methods**: Systematically find all distinct solutions by deflating known ones. [HIGH promise for completeness]

## Key insight

The solution is so smooth (essentially u ≈ 1 + ε·cos(t)) that **2 Fourier modes capture it better than 196+ finite-difference nodes**. The entire scipy optimization campaign (experiments 1-299) was limited by a fundamental discretization floor, not by solver tuning. Switching representations was worth more than 300 experiments of parameter grinding.

## Surprises

- **Expected**: More Fourier modes = more accurate (standard spectral convergence). **Actual**: N=2 beats N=64 by 1400x. **Why**: The solution has almost no energy in higher harmonics. Extra modes add numerical noise through Newton iteration conditioning, not signal.

- **Expected**: n_nodes vs residual should be monotonically decreasing (finer mesh = better). **Actual**: n=196 (1.47e-12) beats n=200 (1.00e-11) and n=300 (3.25e-12). **Why**: Mesh-potential resonance. Specific node counts alias favorably with the cos(t) structure of K(t).

- **Expected**: Bifurcation boundary between trivial and nontrivial basins should be a clean threshold. **Actual**: Chaotic basin structure — u_offset=0.548 (trivial), 0.5485 (nontrivial), 0.549 (crash), 0.5495 (nontrivial). **Why**: Multiple solution branches create fractal basin boundaries in the Newton iteration map.

- **Expected**: Higher-mode perturbations (mode-2, mode-3) would find oscillatory branches with sign changes. **Actual**: All collapse to trivial u=0 or constant u=±1. **Why**: Likely no stable oscillatory branches exist for K_amplitude=0.3. The bifurcation to higher modes may require larger K.

- **Expected**: tol=1e-12 should work if tol=1e-11 works (just tighter). **Actual**: Universal crash for nontrivial branch at tol<4e-12. **Why**: The scipy BVP solver's Newton iteration overshoots when the tolerance demands precision below the discretization error floor — the solver tries to correct noise it can't actually resolve.

## Devil's advocate

**The 2.00e-16 score may be misleading in three ways:**

1. **Different solution**: Fourier N=2 gives solution_norm=1.001298, while N≥3 gives 1.001296. The N=2 solution may be a *different* (less accurate) solution that happens to have low residual. A low residual means the discrete system is satisfied, not that you've found the true continuous solution. With only 2 modes, you're solving a 2-equation system — almost any smooth 2-parameter family can be tuned to near-zero residual of a 2-equation system. **This is the biggest risk.**

2. **Residual vs. solution error**: Residual measures how well the discrete equations are satisfied, not how close the solution is to the true PDE solution. The Fourier N=2 residual is 2.00e-16, but the solution_energy=-1.520921 differs from the N≥4 value of -1.520844. That 0.005% energy difference, while small, suggests the N=2 solution has O(1e-4) solution error despite O(1e-16) residual error. **The metric may reward low-dimensional overfitting.**

3. **The score is genuinely solid IF** the harness evaluates residual of the returned solution on its own high-resolution grid. In that case, N=2 is simply an efficient parameterization of the initial guess, and the residual is computed honestly. But if the harness evaluates residual on the *solver's own grid*, then N=2 is trivially gaming the metric.

**Verdict**: Verify by computing solution error (compare N=2 solution against a high-resolution reference). If the harness uses an independent evaluation grid, the score is legitimate. If not, N=48 (1.35e-13) is the honest best.

## Experiment order

1. **Reproduce winning result** (1 experiment): Fourier N=2, positive branch, confirm 2.00e-16. If it fails, fall back to Fourier N=48 (1.35e-13).
2. **Validate the score** (2 experiments): Run Fourier N=2 solution through independent high-resolution residual check. Compare solution_energy at N=2 vs N=48 vs N=64. If energies diverge, N=2 is gaming.
3. **Newton tolerance sweep at N=2-4** (4-6 experiments): Try newton_tol=1e-13, 1e-14, 1e-15 at N=2,3,4. This is the cheapest unexplored dimension. [HIGH promise]
4. **Continuation/homotopy** (5-10 experiments): Track solution branch as K_amplitude increases from 0 to 0.3. May find bifurcations to new branches.
5. **Deflation** (5 experiments): Use known solutions to deflate and search for undiscovered branches.
6. **Only then**: Return to scipy for any branches that Fourier can't reach.

Total budget to beat or validate current best: ~15-20 experiments.
