# Desires & Feature Requests — agent1

## High Priority

1. **Multi-objective scoring** 
   - Current: only residual matters (marks non-trivial as "discard")
   - Desire: Track (residual, branch_outcome, basin_characterization) separately
   - Why: Chaotic basin mapping is MORE valuable than residual optimization. Non-trivial branches with 3.25e-12 are worth keeping.
   - Implementation: new scoring mode like `design_type=basin_mapping` that doesn't compare to global best

2. **Automated basin boundary search**
   - Current: Manual sweep of u_offset ∈ [0.52, 0.58] by agents
   - Desire: Harness-level parameter sweep (template config with ranges)
   - Why: Phase diagram (u_offset × solver_tol) has ~50-100 cells; need systematic sampling
   - Implementation: `bash run.sh sweep "range" u_offset:[0.52:0.58:0.01] solver_tol:[1e-10:1e-9:0.1]`

3. **Branch tracking in scoring**
   - Current: Only branch identification via solution_mean
   - Desire: Explicit tracking of (branch, residual, basin_width, boundary_distance)
   - Why: For fractal analysis, need to know where basin boundaries are, not just which branch converged

## Medium Priority

4. **Fourier spectral solver comparison**
   - Current: Scipy solve_bvp (4th-order algebraic)
   - Desire: Re-run key experiments with Fourier spectral (exponential convergence claimed)
   - Why: Calibration mentions Fourier achieves 5.55e-17 on non-trivial; worth verifying
   - Method: Add `method: fourier` + `fourier_modes: 64` to config.yaml variants

5. **Bifurcation parameter continuation**
   - Current: K_amplitude=0.3 fixed; only u_offset varies
   - Desire: Parameter continuation in K_amplitude to trace branch connections
   - Why: Understand how fractality emerges as K_amplitude increases
   - Implementation: New domain variant with K_amplitude as tunable parameter

## Low Priority

6. **Fractal dimension estimation**
   - Current: Manual observation of fractal structure in [0.52, 0.58]
   - Desire: Algorithmic estimation (box-counting dimension or Hausdorff dimension)
   - Why: Quantify the fractal complexity (is it Cantor-set-like? self-similar?)
   - Implementation: Post-hoc analysis script on basin grid

7. **Visualization of basin diagram**
   - Current: Text output in results.tsv
   - Desire: 2D heatmap of (u_offset × solver_tol) → branch_outcome
   - Why: Easier to spot patterns, publish-quality figures
   - Implementation: matplotlib/plotly script in post-analysis

## agent5 Session Desires

1. **Combined solver approach:** Desire scipy warm-start followed by Fourier polish to see if we can beat 5.55e-17. Would require implementing a two-stage solver in solve.py.

2. **Fourier mode optimization under chaos regime:** How do Fourier modes (1-8) perform at chaotic u_offset values (0.53, 0.54, etc.)? Are ultra-low modes still optimal in fractally-sensitive basins?

3. **Deflation method:** As noted in calibration.md, deflation could help find additional branches. Request access to Farrell et al. (2015) implementation or guidance on implementing it.

4. **Chebyshev spectral alternative:** Fourier works for periodic; Chebyshev might offer different numerical properties. Not urgent, but worth mentioning.

## agent7 Desires & Future Directions

1. **Explore resonance peaks with Fourier precision**
   - Agent6 found trivial peaks at u_offset≈±0.530 (residual=1.97e-19) and ±0.560 (residual=4.38e-17)
   - Agent2 found non-trivial peaks at u_offset≈±0.889 (residual=5.55e-17)
   - Map full landscape with Fourier 1-mode to find additional peaks/valleys
   - Use to understand bifurcation topography

2. **Fine bifurcation point detection**
   - Current sweep found transition at u_offset≈0.605
   - Fine sweep (Δ=0.001) would pin down exact bifurcation point
   - Compare to theoretical predictions from continuation methods

3. **3D phase diagram: (u_offset, solver_tol, solver_backend)**
   - Agent0 showed tolerance is a bifurcation parameter for scipy
   - Fourier showed robustness across tolerance range
   - Full 3D mapping could reveal whether this is fundamental or solver-specific

4. **Perturbation control validation under Fourier**
   - Agent4 showed phase/amplitude can steer basin selection
   - Test with Fourier 1-mode to confirm control persists or if it's scipy artifact

5. **Parameter space extension**
   - Current work all at K_amplitude=0.3, K_frequency=1
   - Vary K_amplitude ∈ [0.1, 0.5] and K_frequency ∈ [1, 3]
   - See how bifurcation structure changes
   - Look for new branches or codimension-2 bifurcations

6. **Energy-based approach**
   - Current work minimizes BVP residual
   - Try minimizing Hamiltonian energy directly
   - Compare solutions and residuals to see if different stationary points exist
