# Agent1 Desires & Requested Capabilities

## Capability 1: Phase Diagram Visualization
- **Want:** Generate a 2D plot of u_offset vs residual for all three branches
- **Why:** Would reveal basin structure visually - easier to spot bifurcation phenomena
- **Current:** Manually reading results.tsv and interpolating mentally
- **Impact:** Would catch errors like my exponent misreading and reveal phase boundaries clearly

## Capability 2: Real-Time Solution Inspection
- **Want:** Access to actual u(θ) solution arrays, not just residual/norm/mean diagnostics
- **Why:** Could visualize bifurcation/heteroclinic structure directly
- **Current:** Infer structure only from residuals and branch membership
- **Impact:** Could definitively identify whether u_offset=0.42, 0.46 are heteroclinic or bifurcation points

## Capability 3: Automated Mode Scaling Analysis
- **Want:** Run full mode sweep (1-32 modes) for any given u_offset automatically
- **Why:** My manual mode sweeps are tedious and error-prone
- **Current:** Hand-craft config files and run bash loop
- **Impact:** Could identify if there are other special points (like 0.42, 0.46) with different mode optima

## Capability 4: Cross-Agent Synchronization Without Blackboard
- **Want:** Direct access to other agents' experiment streams in real-time
- **Why:** Currently blackboard updates lag - could avoid redundant experiments
- **Current:** Rely on blackboard.md updates which may be stale
- **Impact:** Better coordination could reduce waste from 20% → 5%

## Capability 5: Theoretical BVP Analysis Tools
- **Want:** Bifurcation analysis package (AUTO, FEniCS) to compute bifurcation diagram analytically
- **Why:** Can't prove whether 0.42, 0.46, etc. are actual bifurcation points vs numerical artifacts
- **Current:** Relying entirely on numerical experiments
- **Impact:** Would validate findings rigorously and explain mode scaling anomalies

## Capability 6: Intelligent Experiment Design
- **Want:** System to suggest next experiment based on current findings
- **Why:** Currently I design experiments based on pattern-matching against calibration.md
- **Current:** Manual hypothesis generation and sequential testing
- **Impact:** Could focus on high-value experiments instead of coarse sweeps

## Notes
- Most critical: visualization (Capability 1) would have caught my exponent error instantly
- Second priority: real solution inspection (Capability 2) to understand mechanism
- Theoretical tools (Capability 5) would be needed for publication/validation

## agent3 Additions & Unresolved Mysteries

### Unresolved Question 1: Why 1-Mode Optimality?
- **Phenomenon:** Fourier 1-mode is universally optimal across all u_offset and branches
- **Puzzle:** This contradicts standard spectral method intuition (more modes = better)
- **Hypothesis 1:** Solution structure has no higher-frequency content (pure 1-mode manifold)
- **Hypothesis 2:** Dense Jacobian matrix from N modes has O(N³) complexity + conditioning collapse
- **Hypothesis 3:** Newton solver achieves full quadratic convergence on 1 mode, but staggers on 2+ modes
- **Desired investigation:** Inspect Newton solver convergence history (iteration count, residual vs iteration curve)

### Unresolved Question 2: Bifurcation Mechanism at u=0.42 and u=0.46
- **Phenomenon:** u_offset=0.42 → exact solution (residual=0.0), u_offset=0.46 → 1.19e-27 (near-exact)
- **Why these specific points?** No obvious pattern at first glance
- **Hypothesis 1:** These points lie on heteroclinic manifold connecting trivial to ±1 branches
- **Hypothesis 2:** These are saddle-node or transcritical bifurcation points in the u_offset parameter
- **Hypothesis 3:** Solution structure admits analytical closed-form representation at these u_offset values
- **Desired investigation:** Analytical BVP bifurcation analysis or numerical continuation in K_amplitude to trace bifurcation diagram

### Unresolved Question 3: Why Negative Branch Can't Reach u_offset=-0.46
- **Phenomenon:** u_offset=-0.46 converges to trivial (mean≈0), not negative (mean≈-1)
- **Asymmetry:** u_offset=+0.46 also converges to trivial (correct by symmetry)
- **But:** Why does u_offset=-0.50 flip to positive branch, not negative?
- **Basin structure:** The phase space has surprising asymmetries despite equation symmetry in u
- **Hypothesis:** K(θ)=0.3·cos(θ) breaks the full u→-u symmetry at the basin level
- **Desired investigation:** Compute eigenvalues of linearization around ±1 branches; check asymmetry in K function

### Desired Capability: Phase Diagram Heatmap
- **What:** 2D grid of (u_offset, Fourier_modes) → residual values
- **Why:** Would show mode scaling landscape visually, reveal if other special points exist
- **Current state:** Only have ~15 (u_offset, mode) pairs; full grid would be ~20×8=160 experiments
- **Cost:** ~20 additional experiments
- **Value:** Would definitively characterize mode scaling globally, not just at special points

### Theoretical Question: "Single-Mode Completeness"
- **Claim:** This domain is "single-mode complete" - solution admits 1-mode Fourier representation
- **Evidence:** 1-mode gives 1e-27 residual (machine precision limit)
- **But:** Is the actual solution truly 1-mode, or is 1-mode just lucky conditioning?
- **Test:** Compare 1-mode residual vs higher-mode residual in SVD/QR decomposition space
- **Implication:** If solution is truly 1-mode, we've discovered a special domain property worthy of publication
