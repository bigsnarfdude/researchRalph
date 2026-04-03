# DESIRES — nirenberg-1d-chaos-haiku-h3-4agent-25

## Solver/Implementation

1. **Deflation capability** — Implement Farrell et al. (2015) deflated Newton to systematically eliminate known solutions and search for new branches. Current approach exhausts at 3 branches; deflation could confirm that's a complete enumeration.

2. **Hybrid two-stage solver** — Interface scipy and Fourier solvers in sequence: scipy (coarse, n=100, tol=1e-8) → warm-start with Fourier Newton (1 mode, 1e-12). Could improve convergence on basin boundaries.

3. **Adaptive mode selection** — Auto-detect optimal Fourier mode count per u_offset. At u_offset=0.0 (trivial), 64 modes great; at u_offset=0.9 (non-trivial), 1 mode best. Rule-based selector could save computation.

4. **Lower Newton tolerance** — Current limit is newton_tol=1e-12 (crashes at 1e-13). Needs better preconditioner or damped Newton to push toward 1e-14+.

## Physics / Mathematics

5. **Bifurcation diagram** — Vary K_amplitude ∈ [0, 0.5, 1.0, 1.5] and trace how branches birth, annihilate, or exchange stability. Current K_amplitude=0.3 is fixed; continuation would reveal the underlying geometry.

6. **Continuation in K_frequency** — Move from K_frequency=1 (current) to K_frequency=2, 3, etc. Does the problem admit higher-frequency forcing? Do new branches appear?

7. **Symmetry breaking / asymmetry injection** — Current domain is symmetric under u → -u and θ → -θ+π. Break symmetry and see if new solution families exist.

## Analysis / Diagnostics

8. **Energy landscape visualization** — Plot solution energy E[u] across all (u_offset, u_final_mean) pairs discovered. Visualize energy barriers between branches.

9. **Jacobian conditioning analysis** — Compute spectral condition number κ(J) for each (u_offset, fourier_modes) pair. Correlate κ with observed Newton step sizes and final residuals. Explains why 1-mode beats 64-mode.

10. **Basin boundary fractal dimension** — Compute box dimension of the u_offset ∈ [0.4, 0.7] basin boundary discovered in Phase 2. Is it truly fractal (d < 1) or just complicated?

## Experiments to Run

11. **Higher K_amplitude sweep** — Test K_amplitude ∈ {0.5, 1.0, 1.5} with same Fourier 1-mode config. Do the three branches persist or merge?

12. **Multi-parameter CG descent** — Treat (u_offset, amplitude, phase, n_mode, fourier_modes) as optimization variables and do gradient descent on residual (if gradients computable via AD).

13. **Bifurcation parameter study** — Automated continuation in K_amplitude to find critical bifurcation points (fold, pitchfork, transcritical).
