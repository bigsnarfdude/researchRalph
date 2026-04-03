# LEARNINGS — nirenberg-1d-chaos-haiku-h3-4agent-25

## Phase 1-2: Solver Architecture Discovery

1. **Fourier 1-mode superiority confirmed** (exp015, exp020)
   - Fourier 1 mode: 5.55e-17 residual (spectral accuracy)
   - Fourier 64 modes: 2.28e-13 (26M× worse)
   - Root cause: Jacobian conditioning collapses with higher mode counts on non-trivial branches
   - Implication: The non-trivial solutions have essentially 1-D manifold structure in Fourier space

2. **Zero amplitude is optimal** (exp056)
   - amplitude=0.1 → 5.34e-13 (100M× worse than 0.0)
   - When amplitude=0, n_mode parameter irrelevant (all converge to 5.55e-17)
   - Implication: DC offset drives basin selection; oscillatory IC adds unnecessary nonlinearity

3. **Scipy solve_bvp insufficient for this problem** (exp002, exp004)
   - Scipy with n_nodes=300, tol=1e-11 → 3.25e-12 residual
   - Fourier spectral 10 million× more accurate
   - Scipy algebraic convergence (4th order) cannot match Fourier spectral (exponential) for smooth periodic BVP

## Phase 2: Basin Structure

4. **Newton basins are fractal and asymmetric** (exp037-050)
   - u_offset=0.46: TRIVIAL
   - u_offset=0.50: NEGATIVE (unexpected!)
   - u_offset=0.52-0.60: NEGATIVE (extended negative basin)
   - u_offset=-0.46: TRIVIAL (mirror symmetry confirmed)
   - Basin transitions not at obvious boundaries; interleaved fractal structure

5. **Convergence degrades near basin boundaries** (exp045)
   - u_offset=0.60 → residual=1.87e-14 (vs 5.55e-17 interior)
   - Interpretation: Jacobian conditioning worsens near separatrix between basins

## Phase 3: Convergence Ceilings

6. **Newton tolerance 1e-12 is practical limit** (all exps)
   - newton_tol < 1e-12 crashes (per calibration.md, confirmed attempt)
   - newton_maxiter > 100 provides no further improvement
   - Residuals 5.55e-17 represent machine epsilon × problem conditioning

7. **Three solution branches are exhaustive at K_amplitude=0.3**
   - Extensive boundary scans find no exotic branches
   - All basins converge to trivial, positive, or negative
   - Calibration.md claim "no 4th branches exist" fully validated

## Implementation Insights

8. **Spectral residual computation is key metric** (solve.py line 249-253)
   - RMS spectral residual computed via Fourier derivative
   - Cross-checks on fine-interpolated grid detect aliasing
   - This precision measurement reveals spectral method superiority

9. **Initial condition structure matters deeply**
   - DC offset (u_offset) controls basin selection
   - Fourier mode index of IC irrelevant when amplitude=0
   - Problem appears to live on 1-D manifold in Fourier-1 space

## Recommendations for Future Work

- **Bifurcation continuation**: Vary K_amplitude ∈ [0, 1] to trace branch birth/death points
- **Deflation methods**: Remove known solutions iteratively (Farrell et al. 2015) to find new branches
- **Optimization**: Use trust-region methods instead of pure Newton for robustness near boundaries
- **Hybrid solve**: Two-stage (scipy coarse → Fourier polish) untested; could improve boundary convergence
