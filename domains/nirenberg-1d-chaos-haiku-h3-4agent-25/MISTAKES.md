# MISTAKES — nirenberg-1d-chaos-haiku-h3-4agent-25

## Exp056: Amplitude perturbation strategy failed
**What:** Tried amplitude=0.1 to "enrich" the initial condition and potentially find better convergence
**Result:** Residual degraded to 5.34e-13 (100M× worse than 0.0)
**Lesson:** Added nonlinearity in IC hurts, not helps. Zero oscillation is the natural basin attractor.

## Initial assumption: Fourier modes ≥ 64 better
**What:** Intuition that more Fourier modes = higher resolution = lower residual
**Result:** Fourier 1 mode best (5.55e-17), Fourier 64 modes worst (2.28e-13)
**Lesson:** Read calibration.md before running; this finding was already documented.

## u_offset search: Expected smooth basins
**What:** Assumed basin transitions would be smooth/continuous (u_offset 0.0 → trivial, 0.9 → positive)
**Result:** u_offset 0.46 → trivial, 0.50 → negative (fractal interleaving!)
**Lesson:** Newton method basins can be wildly non-obvious; only empirical mapping works.

## No breakthroughs from parameter sweeps
**What:** Hoped that n_mode index, phase shifts, or extra maxiter would unlock better residuals
**Result:** All attempts converged to same 5.55e-17 or 0.0
**Lesson:** Spectral method + Fourier 1 already at precision ceiling (Newton tolerance 1e-12 limit).
