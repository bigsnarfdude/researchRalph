# LEARNINGS.md

## agent1
- **Fourier spectral method is the key lever**: Switching from scipy (default) to method=fourier with 64 modes improved residuals from ~1e-9 to ~1e-13 on non-trivial branches, and from ~1e-11 to ~1e-21 on trivial branch.
- Config params: method=fourier, fourier_modes=64, newton_tol=1e-12, newton_maxiter=100
- newton_tol=1e-14 with maxiter=50 fails to converge (gets stuck at ~5e-13). Loosening to 1e-12 with maxiter=100 works.
- Positive and negative branches have symmetric residuals (~3e-13 each), confirming u→-u symmetry.
- Trivial branch converges to machine precision (2.8e-21) — essentially exact zero solution.
