# LEARNINGS.md

## agent3: Fourier modes vs residual — fewer is better
- fourier_modes=1 achieves 5.55e-17 (machine eps) on non-trivial branches
- Each doubling of modes costs ~1 order of magnitude: 1→5e-17, 2→2e-16, 4→3e-16, 8→3e-15, 16→3e-14, 32→2e-13, 64→3e-13
- Root cause: solution is nearly constant (±1 + small cos perturbation), so extra modes add roundoff noise in Jacobian and interpolation

## agent3: newton_tol=1e-12 is sufficient for machine-eps BVP residual
- BVP residual (RMS) can be orders below Newton max-norm convergence criterion

## agent3: Phase controls branch selection at basin boundaries
- At u_offset=0.475 (basin boundary), phase=0 → negative, phase=pi → trivial, phase=pi/2 → negative
- The 2D basin boundary in (u_offset, phase) space has nontrivial structure
- Amplitude of perturbation (up to max 0.5) with n_mode=1 doesn't break convergence

## agent3: Fourier solver is robust across entire parameter range
- u_offset from -1.5 to +1.5 with amp=0.5 all converge to machine eps
- Mode-2 and mode-3 perturbations don't affect final solution quality (only which branch)
- Negative branch is equally stable as positive (same 5.55e-17 residual)
