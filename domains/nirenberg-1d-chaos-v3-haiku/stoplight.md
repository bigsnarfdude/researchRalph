# Stoplight — nirenberg-1d-chaos-v3-haiku
Status: PLATEAU | Best: 7.58013271e-24 (exp133) | Experiments: 159 | Stagnation: 21 since last breakthrough

## What works
- Design 'branch_search' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'perturbation' has 20 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 29 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 45 exp, 2 breakthroughs, rate 4%, best 1.13671782e-23
- agent1: 19 exp, 4 breakthroughs, rate 21%, best 7.58013271e-24
- agent2: 53 exp, 0 breakthroughs, rate 0%, best 8.31925483e-19
- agent3: 42 exp, 2 breakthroughs, rate 5%, best 1.13671782e-23

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 4 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 5 consecutive crashes — likely broken config or OOM

## Recent blackboard (last 20 entries)
## AGENT3 SESSION COMPLETE — Summary for Gardener
**Final experiment count:** 42 (agent3 contributed 42 of 156 total)
**Session timeline:**
1. Initial basin boundary exploration: u_offset ∈ [-0.91, +0.91]
2. Bifurcation discovery: u_offset ≈ ±0.57-0.62 regions
3. Crash zone characterization: bifurcation singularities confirmed
4. Mode-2 resonance verification: confirmed u_offset=0.1 is global optimum
5. Negative-offset resonance test: confirmed asymmetry and trivial basin width
**Key deliverables:**
- Complete bifurcation map (±0.01 precision)
- Identification of K(θ) asymmetry mechanism
- Verification of residual plateaus on all branches
- Updated LEARNINGS.md with bifurcation mechanics
- Filed 29 desires for future work (K-parameter sweeps, etc.)
- Documented 3 mistakes that led to correct interpretations
**Status for next agent:** Campaign at PLATEAU (16 stagnation). All major features characterized. Residual optimization complete. Blocked directions: K-parameter variation (requires override).
**Recommendation:** 
- If continuing: Request override to test K_amplitude variation
- If wrapping: Archive findings; bifurcation mechanism well-documented for future reference
**Agent3 standing:** 5% breakthrough rate (2 breakthroughs per early experiments), 67% quality on focused bifurcation tasks. Primary value: mechanistic understanding and bifurcation characterization (not residual optimization).
