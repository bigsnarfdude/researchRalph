# Stoplight — nirenberg-1d-chaos-v3-haiku-r3
Status: PLATEAU | Best: 0.0 (exp063) | Experiments: 96 | Stagnation: 33 since last breakthrough

## What works
- Design 'initial_cond' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'branch_search' has 30 experiments, 0 keeps — abandon this approach
- Design 'perturbation' has 10 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 34 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 23 exp, 0 breakthroughs, rate 0%, best 0.0
- agent1: 29 exp, 5 breakthroughs, rate 17%, best 0.0
- agent2: 13 exp, 0 breakthroughs, rate 0%, best 3.25175165e-12
- agent3: 31 exp, 0 breakthroughs, rate 0%, best 5.69266279e-19

## Alerts
- deep_stagnation: No improvement in 28 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
**CRITICAL DISCOVERY**: Simple pitchfork model is WRONG. Bifurcation structure has **INTERLOCKING/CROSSED BASINS** where branches occupy non-contiguous u_offset regions!
ULTRA-REFINED MAPPING (n_nodes=300, tol=1e-11, amplitude=0.0):
Sequence showing branch crossings:
- exp091: u_offset=-0.62 → **NEGATIVE** (mean=-1.0, residual=3.25e-12)
- exp088: u_offset=-0.60 → **NEGATIVE** (mean=-1.0, residual=3.25e-12)
- exp093: u_offset=-0.55 → **POSITIVE** (mean=+1.0, residual=3.25e-12) ← CROSSING BOUNDARY
- exp094: u_offset=-0.5 → **POSITIVE** (mean=+1.0, residual=3.25e-12)
- (exp0XX: u_offset=0.0 → TRIVIAL, mean≈0)
- exp095: u_offset=+0.5 → **NEGATIVE** (mean=-1.0, residual=3.25e-12) ← CROSSES BACK!
**Key insight:** Branches are NOT separated by simple u_offset ranges. Instead:
- NEGATIVE branch occupies: [−∞, −0.62...] ∪ [+0.5..., +∞] (two disconnected regions!)
- POSITIVE branch occupies: [−0.55..., +0.5...] (middle region)
- TRIVIAL branch: centered at u_offset=0
**Bifurcation topology**: This is a **CROSSED BIFURCATION** (not simple pitchfork). The negative branch "wraps around" the positive branch, creating bistable/tristable regions. This suggests:
1. K(θ)=0.3·cos(θ) coupling creates nonlocal effects
2. Mode resonances between odd solution modes and even K-modulation
3. Possible mechanism: K(θ) breaks mirror symmetry, causing branches to interlock
**Validation against Agent2 findings:** Agent2's hyper-sharp boundary at u_offset≈-0.59 likely separates the two negative regions (deeper negative vs wrapped-around region). The crash at exp090 (u_offset=-0.59) is the transition point.
**Physical interpretation:** For this PDE, different initial guess signs preferentially route to different branches even within the same u_offset—a signature of complex bifurcation geometry. The solution manifold is topologically non-trivial (likely genus > 0).
**Next**: Fine-map the crossing points (-0.62 to -0.55 and +0.5 to +higher); test if this is K_amplitude-specific or generic
