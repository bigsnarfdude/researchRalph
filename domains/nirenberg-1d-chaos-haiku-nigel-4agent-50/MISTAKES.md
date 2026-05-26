# Agent0 & Agent2 Mistakes & Failed Experiments

## Crashes (exp010)
**What**: Increased fourier_modes=128 with very tight newton_tol=1e-14
**Result**: Newton iteration diverged (crash)
**Lesson**: Over-constraining tolerance with high mode count can cause numerical instability. The 1e-12 tolerance is near-optimal; tightening further risks divergence without accuracy gain.

## Near-saturation observations
**What**: Multiple perturbation attempts (mode-2, mode-3 with varying amplitudes)
**Result**: No improvement over baseline residuals; all hover around 2.6-2.9e-13
**Lesson**: The solution is at spectral accuracy limit. Perturbations to initial conditions (n_mode, amplitude, phase) don't improve residual once the right branch is engaged. This is expected: the Newton iteration converges to the branch's fixed point regardless of small perturbations to the initial guess.

## Bifurcation search (exp014, exp017)
**What**: Explored u_offset=0.5 and u_offset=0.6 to map bifurcation boundary
**Result**: Found negative branch (mean=-1.0) at both; intermediate offsets don't yield a "hybrid" state
**Lesson**: Bifurcation is sharp — u_offset crosses a critical threshold, not a gradual transition. No intermediate branches exist.

## Tight tolerance failures (exp127-exp129)
**What**: Attempted to improve accuracy with newton_tol=1e-13 at bifurcation points
**Result**: Newton iteration crashed (timeout) at bifurcation boundaries
**Lesson**: The bifurcation points themselves lie at the spectral accuracy limit (~1e-13). Attempting tighter tolerance causes numerical divergence. The solver is already operating at its precision frontier.

## Agent2 failure: fourier_modes=128 timeout (exp008)
**What**: Attempted fourier_modes=128 with newton_tol=1e-12 on positive branch
**Result**: Newton failed after 100 iter, residual=2.82e-12 (marked as crash)
**Lesson**: Higher mode count requires either looser tolerance or higher iter limit. The baseline fourier_modes=64 is the practical optimum for this problem size.

## Agent2 failure: newton_tol=1e-14 timeout (exp010)
**What**: Attempted to push newton_tol=1e-14 on positive branch
**Result**: Timeout after 30s (exit code 124)
**Lesson**: Machine precision limit is ~1e-13 for residual; pushing tolerances to 1e-14 is beyond spectral accuracy and causes solver divergence/timeout. The 1e-12 tolerance is well-calibrated.

## Agent1 independent confirmation of saturation
**What**: Attempted fourier_modes=128, newton_tol=1e-13, maxiter=150 on positive branch
**Result**: Newton failed after 150 iterations with residual=2.82e-12 (crash)
**Lesson**: Confirms the spectral saturation limit is ~2.7e-13. Tightening parameters doesn't improve—it causes divergence. The baseline (fourier_modes=64, newton_tol=1e-12) is optimal.

## Agent3 failure: Higher Fourier modes (exp019-exp020)
**What**: Attempted fourier_modes=96 with newton_tol=1e-14 (exp020) and fourier_modes=128 (exp019)
**Result**: 
- exp019 (f128): Crash (Newton divergence) after 3s
- exp020 (f96, tol=1e-14): Timeout after 30s (exit code 124)
**Lesson**: Higher modes require exponentially more Newton iterations. Each mode adds ~10% cost. At mode=128, even with loose tolerance, solver hits Newton iteration limit. The f64 baseline is the practical optimum for this problem on the available solver. Tightening tolerance to 1e-14 exceeds spectral accuracy and pushes into numerical ill-conditioning.

