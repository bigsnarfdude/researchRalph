# Agent Learnings

## exp001: Trivial branch baseline
- u_offset=0.0 with default solver settings achieves perfect residual (0.0)
- Trivial branch is stable and easily found with zero initial condition
- Fourier spectral method with 64 modes and Newton tolerance 1e-12 works well
- All solver parameters in config.yaml are well-tuned for this problem

## exp002-004: All three solution branches located (agent2)
- exp002: Trivial branch (u_offset=0.0) → residual=0.0, mean=0.0 (confirmed)
- exp003: Positive branch (u_offset=0.9) → residual=2.67e-13, mean=1.000 (high quality solution)
- exp004: Negative branch (u_offset=-0.9) → residual=2.67e-13, mean=-1.000 (high quality solution)
- Solution space is fully explored: all three branches found with near-zero residuals
- Branch selection is purely controlled by u_offset: boundary locations at ±0.9 for ±1 branches
- Both ±1 branches achieve identical residual (2.67e-13) despite opposite signs

## Complete Bifurcation Phase Diagram (agents 1-3)

### Positive u_offset side:
- **u_offset ∈ [0, 0.45]** → trivial branch (mean≈0, residual≈0)
- **u_offset ∈ [0.46, 0.49]** → positive branch (mean≈+1.0, residual ≈2.76e-13)
- **u_offset ∈ [0.50, 0.58]** → **NEGATIVE branch (mean≈-1.0)** ← anomalous! (residual ≈2.13e-13)
- **u_offset ∈ [0.59, 0.9]** → positive branch (mean≈+1.0, residual ≈3.88e-13)

### Negative u_offset side (MIRROR STRUCTURE):
- **u_offset ∈ [-0.45, 0]** → trivial branch (mean≈0)
- **u_offset ∈ [-0.58, -0.45]** → **POSITIVE branch (mean≈+1.0)** ← anomalous! (residual ≈2.13e-13)
- **u_offset ∈ [-0.59, -0.75]** → negative branch (mean≈-1.0, residual ≈2.76e-13)
- **u_offset ∈ [-0.9, -0.75]** → negative branch (mean≈-1.0, residual ≈4.35e-13)

### Key findings:
- **Nonmonotonic bifurcation**: Positive offsets can select negative branch and vice versa
- Amplitude/phase/mode perturbations do NOT change branch identity
- All branches converge to machine precision (≈2.7e-13) — no solver improvement possible
- 128 Fourier modes crash solver on non-trivial branches
- Newton tolerance >1e-12 crashes solver (already at precision limit)

## Extrema testing (agents 1-3, >140 experiments total)
- Tested u_offset extremes: ±1.2, ±2.0 all converge to expected branches
- Robustness confirmed: bifurcation structure stable across parameter ranges
- **CRITICAL DISCOVERY**: 2D bifurcation surface in (u_offset, amplitude) space!
  - u_offset=0.585, amplitude=0.0 → negative branch
  - u_offset=0.585, amplitude=0.05 → positive branch (amplitude acts as branch selector!)
  - Amplitude effect localized to narrow boundary region (~0.58-0.59)
  - Strong effect at bifurcation, weak effect in core regions (0.55 still negative even at amp=0.1)
- **SATURATION**: Bifurcation structure fully characterized
- Best achievable: residual=0.0 (trivial), ~2.7e-13 (both ±1 branches)
- Solver at machine precision limit; further optimization impossible with current method

## exp006-007, exp009-013: Phase diagram mapping (agent3)
- Discovered **complex phase transition behavior** in u_offset parameter:
  - u_offset ≤ 0.45 → trivial branch (mean≈0, residual≈0)
  - u_offset ≈ 0.50-0.55 → **negative branch** (mean≈-1.0) — UNEXPECTED!
  - u_offset ≥ 0.60 → positive branch (mean≈+1.0)
- This reveals **nonmonotonic bifurcation**: positive u_offset can access negative branch
- All branches maintain machine-precision residual (≈2.67e-13)
- Amplitude and mode perturbations do not change branch selection
- 128 Fourier modes cause solver crash (overflow/precision issue)

## exp006-157: Comprehensive boundary refinement (agent3, 50+ exps)
- **Amplitude control at u_offset=0.50 (primary critical point)**:
  - amplitude ≤ 0.14 → negative branch (stable)
  - amplitude ≥ 0.15 → positive branch (stable)
  - Precise critical value: amplitude_crit ≈ 0.142±0.003
- **Negative-positive bifurcation boundary refined**: 0.58975 ± 0.0001
  - Sharp discontinuity (no intermediate solutions across transition)
- **2D scan results**: Amplitude effect localized to u_offset ≈ 0.50 only
  - u_offset = 0.48, 0.52, 0.55 insensitive to amplitude changes
- **K-function parameter sweep**: K_amplitude, K_frequency affect magnitude, NOT branch identity
- **Final assessment**: All parameter spaces exhausted; no new dynamics discoverable
  - Domain saturation confirmed: 159 total experiments, all three branches fully characterized
  - Bifurcation structure intrinsic to PDE; forcing/initial-condition variations only modulate solution magnitude
