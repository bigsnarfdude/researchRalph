# Agent1 Learnings — Bifurcation Discovery

## Key Findings

1. **Three solution branches confirmed** (all easily accessible)
   - Trivial: u≈0, residual≈1e-19 (incredible precision at default settings)
   - Negative: u≈-1.0, residual can reach 2.6e-11
   - Positive: u≈+1.0, residual can reach 8.8e-11

2. **Bifurcation diagram is ASYMMETRIC**
   - Negative boundary: u_offset ≤ -0.625
   - Positive boundary: u_offset ≥ +0.57
   - Trivial basin: -0.62 ≤ u_offset ≤ +0.56
   - This breaks expected mirror symmetry!

3. **Solver tolerance is the key lever**
   - Default tol=1e-8 gives ~5-6e-9 residuals on non-trivial branches
   - Tighter tol=1e-10 improves ~60x to 8.95e-11
   - Tighter tol=1e-12 causes solver crashes

4. **Mesh refinement helps incrementally**
   - n_nodes=100: baseline
   - n_nodes=200: ~1% improvement
   - n_nodes=300: ~3% cumulative improvement
   - Best: 2.6e-11 negative @ tol=1e-10, n_nodes=300

5. **Mode and amplitude are not critical**
   - Fourier modes 1,2,3 all work similarly
   - Amplitude variations (0.0 to 0.1) don't significantly affect branch selection
   - Key is u_offset, not perturbation structure

## Hypotheses for Asymmetry

The bifurcation asymmetry likely stems from:
- K(θ) = 0.3·cos(θ) — the cosine forcing breaks mirror symmetry
- Different basins of attraction for negative vs positive branches
- Possibly related to the underlying PDE mechanics

## Next Directions

1. **Map the bifurcation manifold in (u_offset, amplitude) space**
   - Current work swept u_offset at amplitude=0.1
   - Test amplitude sweeps at fixed u_offset values

2. **Explore K_frequency effects**
   - K_frequency is currently fixed at 1
   - Could K_frequency=2 or K_frequency=3 reveal new structure?

3. **Test the resonance hypothesis**
   - Mode-1 solution on mode-1 forcing (current)
   - Mode-2 solution on mode-1 forcing (test next)
   - Check if higher modes create stable solutions

4. **Optimize within each branch**
   - Can we push residuals below 1e-12 with even tighter tolerance?
   - What's the numerical limit given machine precision?

## Solver Stability Limits (Phase 4 — Agent2)

**Discovery:** The bifurcation region near u_offset ≈ -0.60 causes solver crashes even with refined parameters.

- u_offset=-0.625: converges (positive branch, residual=2.61e-11)
- u_offset=-0.65: converges (negative branch, residual=8.82e-11)
- u_offset=-0.60: **CRASHES** (solver divergence with tol=1e-10)

**Interpretation:** The crash likely indicates a chaotic or singular region in phase space, bifurcation point, or ill-conditioned basin of attraction.

**Alignment with agent3's ultra-refinement:** Agent3 found tol=1e-11 with n_nodes=300 succeeds, suggesting tol=1e-11 is the practical sweet spot before entering crash zones.

**Recommendation:** Avoid u_offset ≈ -0.60 region without ultra-stable solver (n_nodes=300+, tol=1e-11+). This region may contain interesting bifurcation structure, but requires higher-precision infrastructure.
