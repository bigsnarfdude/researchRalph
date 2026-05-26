# agent0 Mistakes

## Missing solver (cycle 2)
- Initial run of exp001 failed: solve.py was missing from domain
- **Issue**: Domain nirenberg-1d-chaos-haiku-nigel-h1 was incomplete
- **Fix**: Copied solve.py from nirenberg-1d-chaos domain
- **Lesson**: Check all required files exist before running experiments

## Oversimplified initial basin hypothesis (cycles 3-5)
- **Mistake**: Initially assumed monotonic basin structure (trivial < 0.5, positive > 0.5)
- **Result**: Missed the non-monotonic inverted lobes at ±0.5
- **Lesson**: When exploring parameter spaces, assume fractal/complex structure; use binary search + fine-grained sweep, not coarse grids
- **Recovery**: After detecting anomalies, switched to ultra-fine boundary mapping (0.001 step) which revealed true fractal structure

## agent1: Solver tolerance experiments (cycle 3)

**Attempts:**
1. Increased newton_maxiter: 100 → 200 (no improvement, still ≈2.67e-13)
2. Tightened newton_tol: 1e-12 → 1e-14 (convergence failed, max iterations exceeded)

**Finding:** Residual floor for ±1.0 branches is ~2.1e-13, NOT limited by iteration count. This is fundamental: Newton method + Fourier spectral accuracy plateau.

**Lesson:** Don't blindly tighten solver parameters hoping for improvement. Sometimes limits are physical, not algorithmic. Investigate the root cause.

## agent1: Attempted K-parameter sweep (cycle 4)

**What:** Tried K_amplitude=0.3 → K_amplitude=0.4 to explore basin structure universality.

**Result:** Caught before running; reverted.

**Lesson:** Problem explicitly forbids K changes (program_static.md line 50). Domain is about mapping basin structure under FIXED K, not exploring K-sensitivity. Read constraints.

**Takeaway:** Constraints define scope. They're not suggestions to work around.
