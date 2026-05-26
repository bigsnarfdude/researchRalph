# Stoplight — nirenberg-1d-chaos-haiku-nigel-h2

**Status**: SOLVED & SATURATED
**Best**: residual=0.0 (trivial branch, u_offset=0.0)
**Experiments**: 156 total | Crashes: 3 | Kept: 3 | Discarded: 150
**Stagnation**: 40+ experiments since last improvement (problem saturated)

## Recent Progress (Last 10 Exp)
- exp147-exp156: Mode and amplitude refinements on positive/negative branches
- All residuals in [2.6e-13, 3.9e-13] range
- No improvement over agent0's best of 2.36e-13

## Critical Observations
1. **Trivial branch**: residual=0.0 (exact, analytically verified)
2. **Non-trivial branches**: residual saturation at 2–3 × 10^-13 (double-precision limit)
3. **Convergence**: All three branches converge stably; no numerical instability observed
4. **Basin coverage**: All three solution branches found and verified

## Recommendation
**Problem is complete for this K(θ).** Further optimization is not feasible without:
- Changing K_amplitude, K_frequency (currently fixed)
- Using higher-precision arithmetic (mpmath)
- Moving to new domain

**Suggest: Archive this domain, move to new PDE or vary K parameters.**
