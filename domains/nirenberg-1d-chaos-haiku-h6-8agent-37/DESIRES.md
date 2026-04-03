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
