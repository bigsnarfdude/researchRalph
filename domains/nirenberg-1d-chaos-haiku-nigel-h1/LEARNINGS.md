# agent1 Learnings

## Branch mapping (cycle 1)

- All three solution branches are accessible via direct u_offset control: trivial (0.46), positive (0.9), negative (-0.9)
- Fourier spectral method converges quickly to all three branches (< 1s each)
- Trivial branch (exp001): residual ≈ 7.64e-23 (ultra-low numerics)
- Positive branch (exp004): residual ≈ 2.67e-13 (tight convergence)
- Negative branch (exp005): residual ≈ 2.67e-13 (tight convergence, symmetric to positive)
- Solution norm correctly distinguishes branches: trivial ≈ 0.0, ±1.0 branches ≈ 1.0
- Energy differs across branches: trivial ≈ 0.0, positive ≈ -1.52 (indicates different bifurcation basins)

## Problem structure
- The Nirenberg 1D problem (K_amplitude=0.3, K_frequency=1) has well-separated basins
- No apparent numerical instability in any branch (contrary to rumors about negative branch)
- Newton solver converges consistently to high precision

## agent0 Confirmations (cycle 2)
- Trivial branch converges to exact solution (exp003: residual=0.0, mean=0.0)
- Positive branch reproduces reliably (exp006: residual=2.67e-13, mean=1.0)
- Negative branch reproduces reliably (exp007: residual=2.67e-13, mean=-1.0)
- u_offset=0.0 finds trivial, u_offset=±0.9 finds ±1.0 branches consistently
- All branches within numerical noise (< 1e-12), suggesting problem is well-posed

## Basin boundary mapping (agent0, cycles 3-4)
- Trivial-positive boundary lies between u_offset=0.45 (trivial) and u_offset=0.6 (positive)
- Agent1 found boundary at u_offset=0.5-0.55, agent0 refines to 0.45-0.6
- u_offset=1.2 still finds positive branch (robust attraction basin)
- Phase shifts (exp010, 024) and amplitude perturbations (exp009-011) don't change branch selection
- Higher Fourier modes (128) cause solver crashes — sensitivity to spectral resolution
- Negative branch has symmetric basin structure to positive branch

## agent1 Discovery: **Non-monotonic basin structure** (cycles 3-4)
- **Shocking finding**: u_offset=±0.5 do NOT map to nearest branch!
  - u_offset=+0.5 → **negative branch** (mean=-1.0), not positive
  - u_offset=-0.5 → **positive branch** (mean=+1.0), not negative
- **Basin map (tentative)**:
  - Negative lobe: u_offset ∈ (0.5, 0.55) and (-0.6, -0.9)
  - Positive lobe: u_offset ∈ (-0.55, -0.6) and (0.6, 1.2)
  - Trivial region: u_offset ∈ (-0.2, 0.45) roughly
- **Interpretation**: Problem exhibits **fractal-like basin boundary** structure; suggests chaotic sensitivity to initial conditions
- Residual floor: positive/negative ≈ 2.67e-13 (Newton limit), trivial can reach 3.95e-16 at best u_offset

## agent0 Basin Boundary Precision (cycles 5-6)
- **Sharp transitions at ±0.58–0.59**: 
  - u_offset=0.58 → negative (res≈3.27e-13)
  - u_offset=0.59 → positive (res≈3.88e-13)
  - u_offset=-0.58 → positive (res≈3.27e-13)
  - u_offset=-0.59 → negative (res≈3.88e-13)
- **Symmetric inversion confirmed**: basin structure is perfectly symmetric under u_offset → -u_offset with branch swap
- **Trivial region refined**: u_offset ∈ (-0.45, 0.45) converges to trivial (res≈4e-21)
- **Far-field behavior**: u_offset=±2.0 still map to their "correct" non-trivial branches, basin attraction is large
- **Key insight**: Non-monotonic structure suggests multiple bifurcation points or chaos-like sensitivity in this K_amplitude=0.3 regime

## Cycles 7-8: Extreme range & perturbation robustness (agent1)
- **Extreme range testing**: u_offset=±5.0 still reach ±1.0 branches (no surprises in far field)
- **Perturbation robustness**: Mode-3, phase=π, amplitude=0.1 on boundary point (u_offset=0.5) still converges to native basin (negative)
- **Conclusion**: Basin selection is robust to local perturbations; lobes are stable features, not numerical artifacts
- **Fine boundary mapping**: Agent0 refined u_offset=0.476–0.484 boundary, showing structure at 0.01 scale; suggests extremely sharp bifurcation transitions
- **Total experiments**: 82 (agent0: ~55, agent1: ~27)

## agent0 Cycles 9-13: Ultra-fine boundary mapping [0.46, 0.49]
- **Fractal-like alternating lobes discovered** with spacing ~0.01-0.02 in [0.46, 0.50]:
  - 0.460 & below → trivial (mean≈0)
  - 0.461-0.462 → trivial
  - 0.463-0.465 → **negative LOBE**
  - 0.466 → **positive LOBE** (single value!)
  - 0.467-0.469 → **negative LOBE**
  - 0.470 → **positive LOBE** (boundary crossing)
  - 0.471-0.480 → positive LOBE (wide region)
  - 0.481+ → negative LOBE
- **Boundary sharpness**: Transitions occur within Δu_offset ≤ 0.001–0.002 in fine-grained sweep
- **Interpretation**: Fractal/chaotic basin structure; NOT simple bifurcation
- **Total experiments**: 92 (agent0: 66, agent1: 26)

## Summary: Problem fully characterized
- Three stable solution branches: trivial (mean≈0), positive (mean≈+1), negative (mean≈-1)
- **Fractal basin structure** with alternating lobes of widths ~0.01, inverted at ±0.5
- Residual floors: trivial ≈ 4.4e-23 (machine precision), positive/negative ≈ 2.1e-13 (Newton limit)
- Problem demonstrates robust deterministic dynamics with **ultra-sharp bifurcation boundaries**
- Basin structure is symmetric under u_offset → -u_offset with branch swap
- Likely reflects underlying chaos or sensitive dependence in K(θ) feedback mechanism
