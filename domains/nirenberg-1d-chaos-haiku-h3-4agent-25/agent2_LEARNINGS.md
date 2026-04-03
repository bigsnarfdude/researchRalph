# Agent2 Learnings (exp005-031)

## Phase 1: All branches verified (exp005-008)
- Trivial: u_offset=0, n=196, tol=1e-12 → residual=0.0 ✓
- Positive: u_offset=0.9, n=300, tol=1e-11 → residual=3.25e-12 ✓
- **Negative: u_offset=-0.9, n=300, tol=1e-11 → residual=3.25e-12 ✓ PERFECTLY SYMMETRIC**

**Key finding:** Negative branch is numerically robust and stable. Symmetry is exact. No solver artifacts.

## Phase 2: Scipy n_nodes sweep on non-trivial branches (exp016-029)

**Sweet spot discovery:** n_nodes = 196-200
| n_nodes | residual | status |
|---------|----------|--------|
| 150 | 7.78e-12 | discard |
| 170 | 7.78e-12 | discard |
| 194 | 1.52e-12 | discard |
| 196 | 1.47e-12 | **BEST** |
| 198 | 1.47e-12 | **BEST** |
| 200 | 1.47e-12 | **BEST** |
| 250 | 5.63e-12 | discard |
| 300 | 3.25e-12 | discard |

**Non-monotonic pattern:** Residual improves 150→196 (5.3x), then degrades 196→250→300 monotonically. This is NOT a smooth optimization surface; there's a sharp optimum at n≈196-200.

## Scipy tighter tolerance fails
- pos_n196_tight: tol=1e-12 on positive branch → **CRASHES** (confirms calibration note)

## Phase 3: Ready for Fourier spectral optimization
- Agent3 reports Fourier 64 modes, newton_tol=1e-12 → residual=2.278e-13 (10x better than baseline scipy)
- My scipy improvement: 3.25e-12 → 1.47e-12 (2.2x)
- Goal: Match or exceed Fourier 2.278e-13 with more aggressive parameter tuning
