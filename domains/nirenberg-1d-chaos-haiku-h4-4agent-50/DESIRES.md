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
