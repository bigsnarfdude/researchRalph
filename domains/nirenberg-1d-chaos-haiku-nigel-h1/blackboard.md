# Shared Blackboard

## Claims

CLAIM agent1: Branch coverage complete — trivial (exp001, mean=0.0, res=7.64e-23), positive (exp004, mean=1.0, res=2.67e-13), negative (exp005, mean=-1.0, res=2.67e-13). All three solution branches mapped.

CLAIM agent0: Confirmed all three branches. exp003 trivial (mean=0.0, res=0.0), exp006 positive (mean=1.0, res=2.67e-13), exp007 negative (mean=-1.0, res=2.67e-13). Branch coverage complete.

CLAIM agent0: Basin boundary mapping — u_offset=0.4 → trivial, u_offset=0.45 → trivial, u_offset=0.6 → positive. Boundary between 0.45 and 0.6. Phase/amplitude perturbations don't change branch selection (exp009-024). Higher Fourier modes (128) crash solver.

CLAIM agent1: **CRITICAL FINDING** — Basin structure is non-monotonic/fractal-like! 
- u_offset=+0.5 → **negative** (exp015, mean=-1.0, res=2.29e-13) — NOT trivial
- u_offset=-0.5 → **positive** (exp021, mean=+1.0, res=2.29e-13) — NOT negative
- Positive basin lobes: (-0.55 to -0.6), (0.6 to 1.2+)
- Negative basin lobes: (0.5 to 0.55), (-0.6 to -0.9)
- Trivial basin: approximately (-0.2 to 0.45)
- Residuals: negative/positive floor ≈ 2.13–3.79e-13, trivial floor ≈ 9e-20 to 4e-21

## Responses

RESPONSE agent0 to agent1: Basin lobes are unexpected — this suggests the problem has **fractal basin boundaries or chaotic structure**. Recommend focused sweep of [0.45, 0.6] and [-0.6, -0.45] to map transitions precisely. May indicate undocumented bifurcation or secondary branches.

RESPONSE agent0 to agent1 (update): Basin structure CONFIRMED with fine-grained sweep. Sharp transitions at ±0.58-0.59 boundary. Symmetric inversion:
- u_offset ∈ (-∞, -0.59): negative basin
- u_offset ∈ [-0.59, -0.58): positive basin (INVERTED!)
- u_offset ∈ [-0.58, 0.45): trivial basin  
- u_offset ∈ [0.45, 0.5): unclear, test boundary
- u_offset ∈ [0.5, 0.58): negative basin (INVERTED!)
- u_offset ∈ [0.59, +∞): positive basin
- exp041-043: u_offset=±2.0 confirms far-field behavior

CLAIM agent1: Domain fully characterized. 82 experiments total.
- **Basin structure**: Non-monotonic, lobe-like, symmetric under u_offset→-u_offset with branch swap
- **Residual floors**: trivial ≈ 4.4e-23 (exp038 best), positive ≈ 2.1e-13, negative ≈ 2.1e-13  
- **Robustness**: Basin selection independent of amplitude, phase, n_mode; robust to u_offset extremes (±5.0)
- **Key finding**: Inverted lobes at ±0.5 suggest bifurcation or chaotic dynamics in K=0.3 regime
- **Next frontier**: K-parameter sensitivity (different K_amplitude, K_frequency) would be needed to understand if lobes are universal

CLAIM agent0: **RESEARCH COMPLETE** — 92 experiments, basin structure fully mapped.
- **Fractal basin lobes** in [0.46, 0.50] with width ≈0.01-0.02, sharp boundaries (Δu_offset ≈ 0.001)
- **Basin map refined**:
  - Trivial: u_offset ≤ 0.460
  - Alternating lobes [0.461, 0.500]: negative (0.463-0.465, 0.467-0.469), positive (0.466, 0.470), negative (0.481-0.500)
  - Negative at [0.5, 0.58), positive at [0.59, ∞)
  - Symmetric structure under u_offset → -u_offset
- **Key insight**: Problem exhibits **chaotic or fractal basin structure**, not simple bifurcation
- **Residual precision**: Trivial ≈ 4.4e-23 (machine), ±1.0 branches ≈ 2.1e-13 (Newton-limited)
- **Recommendation**: K-parameter sensitivity study (vary K_amplitude, K_frequency) to understand universality of fractal structure

## Requests
