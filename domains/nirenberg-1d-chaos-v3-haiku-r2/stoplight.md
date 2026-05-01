# Stoplight — nirenberg-1d-chaos-v3-haiku-r2
Status: STAGNANT | Best: 1.10623942e-22 (exp076) | Experiments: 456 | Stagnation: 379 since last breakthrough

## What works
- Design 'solver_param' produced 2 breakthroughs — double down here

## Gaps — unexplored
- 13 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 47 exp, 1 breakthroughs, rate 2%, best 5.57300013e-20
- agent1: 105 exp, 2 breakthroughs, rate 2%, best 2.47475744e-19
- agent2: 254 exp, 2 breakthroughs, rate 1%, best 1.10623942e-22
- agent3: 50 exp, 0 breakthroughs, rate 0%, best 6.54666394e-18

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 4 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM

## Recent blackboard (last 20 entries)
   - u_offset=0.50: mode1→trivial, mode2→positive, mode3→trivial
   - u_offset=0.55: mode1→negative, mode2→trivial, mode3→trivial
   - **Reversal**: mode behaviors are u_offset-dependent!
4. **Bifurcation is codimension-3**: (u_offset, amplitude, mode) fully specify basin access
   - NOT codimension-2 as earlier thought
   - Phase is orthogonal (no effect on branch selection)
5. **Solution space is stratified**:
   - Different modes access different subsets of three branches
   - No single mode can access all basins from any u_offset
   - K(θ) resonance (mode-1) likely drives differentiation
**Publication readiness**: HIGH
- Sophisticated bifurcation structure
- Mode-dependent basin architecture (novel)
- Codimension-3 manifold with resonance structure
- Potentially significant for nonlinear dynamics / bifurcation theory
**Recommended next step**: MATHEMATICAL ANALYSIS + VISUALIZATION
- Before more experiments, analyze theoretical structure
- Floquet/perturbation theory on mode coupling
- 3D bifurcation diagram visualization
**Status**: RESEARCH CONTINUES. New discoveries suggest deeper structure awaiting investigation.
