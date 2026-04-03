# Mistakes — agent1, nirenberg-1d-chaos-haiku

## Early Solver Crashes (exp003, 005, 006)

**What:** Attempted positive branch (u_offset=0.8-0.9) with u_offset jump from 0.0, causing solver to request mesh beyond budget.

**Result:** Three consecutive crashes with error "maximum number of mesh nodes exceeded"

**Lesson:** Non-trivial branches (large u_offset jumps) require careful solver tuning. Always increase n_nodes to near-max (300) before attempting large offsets. Amplitude matters too—starting with amplitude=0.1 was too aggressive.

**Corrected approach:** Conservative amplitude (0.0), maxed n_nodes (300), tight tolerance (1e-11) works reliably for ±0.9 offsets.

## Tolerance Paradox (exp012)

**What:** Tried to improve residuals by tightening tolerance from 1e-11 to 1e-12 on positive branch.

**Result:** Solver crashed immediately despite identical mesh config.

**Lesson:** For this problem, 1e-11 is the optimal tolerance sweet spot. Tighter 1e-12 causes internal bifurcation detection or adaptive mesh exhaustion. This likely reflects the adaptive mesh algorithm in scipy.integrate.solve_bvp.

**Implication:** Residuals capped at 1e-12 for nontrivial branches under current solver config. Exceeding this limit requires backend change (Fourier spectral method?).

## Branch Assumption Error (exp017)

**What:** Assumed u_offset=0.5 (midpoint) would interpolate between trivial and positive branches.

**Result:** Unexpectedly landed on negative branch (mean=-1.0), not interpolation.

**Lesson:** Solution branches are discrete attractors, not continuously interpolating. The midpoint initialization selects the nearest basin—which turns out to be negative for this problem geometry.

**Implication:** Basin boundaries are nonsmooth (likely fractal). Explore them carefully with small steps.

---

# Mistakes — agent0, nirenberg-1d-chaos-haiku

## Over-discretization in n_nodes (exp018-019, 022)

**What:** Thought finer initial mesh (n=210, 220, 300) would improve convergence for scipy.

**Result:** All worse than n=196. Residual degraded from 1.47e-12 → 9.51e-12.

**Lesson:** scipy.integrate.solve_bvp has internal complexity; finer initial grids trigger different adaptive refinement, often suboptimal. n=196 is a sweet spot balancing initial DOF and adaptive mesh growth.

**Recovery:** Stick with n≈196 for scipy non-trivial.

## Tolerance escalation on scipy (exp011, 013, 016)

**What:** Tested three tolerance modifications: looser (1e-10), intermediate (5e-12), tighter (1e-12).

**Result:**
- 1e-10: residual=4.50e-11 (WORSE)
- 5e-12: residual=4.99e-12 (slightly worse)
- 1e-12: CRASH

**Lesson:** 1e-11 is the optimal tolerance for scipy on non-trivial branches. Looser relaxes solver strategy. Tighter exhausts DOF budget.

**Implication:** scipy ceiling at ~1-3e-12 residual is not precision-limited but DOF-limited.

## Fourier modes 2-4 assumption (exp030-032)

**What:** Expected more Fourier modes → better spectral approximation.

**Result:** All degraded: 2-mode 2.00e-16, 3-mode 4.42e-16, 4-mode 2.57e-16 (vs 1-mode 5.55e-17).

**Lesson:** Solution has minimal Fourier support. The solution is nearly 1-mode dominant. Adding modes introduces conditioning errors from the O(M³) dense Jacobian in Fourier Newton method.

**Recovery:** Use fourier_modes=1 exclusively.
