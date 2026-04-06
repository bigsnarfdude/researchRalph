# Stoplight — nirenberg-1d-chaos-v3
Status: STAGNANT | Best: 0.0 (exp026) | Experiments: 156 | Stagnation: 130 since last breakthrough

## What works
- Design 'initial_cond' produced 2 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'branch_search' has 94 experiments, 0 keeps — abandon this approach
- Design 'perturbation' has 9 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 19 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 37 exp, 1 breakthroughs, rate 3%, best 0.0
- agent1: 49 exp, 3 breakthroughs, rate 6%, best 9.25602021e-29
- agent2: 44 exp, 0 breakthroughs, rate 0%, best 2.64660147e-24
- agent3: 26 exp, 0 breakthroughs, rate 0%, best 1.28666048e-23

## Alerts
- deep_stagnation: No improvement in 125 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
  Confirmed at 4, 16, and 32 Fourier modes — norm is STABLE at 0.070536.
  This is NOT the trivial branch (norm=0.000000). It's a small-amplitude oscillatory solution with zero mean.
  All teams previously stated only 3 branches exist — this may be a 4th genuine solution.
NOTE agent1: Phase=π/2 is the key IC for accessing this branch. Phase=0 → positive, phase=π → trivial. Phase selects the basin.
CLAIM agent1: Phase is a basin selector at u_offset=0.45 (Fourier, 4 modes):
  phase=0    → positive branch (norm=1.001, mean=+1.000)
  phase=π/2  → ANOMALOUS branch (norm=0.071, mean=0.000) ★ NEW
  phase=π    → trivial (norm=0.000, mean=0.000)
CLAIM agent1: VERIFICATION — Scipy with identical IC (offset=0.45, amp=0.1, phase=π/2) gives norm=0.000 (trivial). The anomalous norm=0.071 solution is ONLY found by Fourier solver. However:
  - Stable across 4, 16, 32 Fourier modes (always 0.070536)
  - Residual is machine epsilon (1.86e-16)
  - BVP equation is satisfied to float64 precision
  INTERPRETATION: This is likely a genuine but UNSTABLE equilibrium that scipy's shooting method cannot find (it always falls to the stable trivial attractor). The Fourier Newton method, being a global spectral approach, can find unstable fixed points. This would be a saddle-type solution between trivial and ±1 branches.
CLAIM agent1: 4th branch is ROBUST — found at multiple u_offsets with phase=π/2:
  u_offset=0.0:  norm=0.070535, residual=7.08e-17
  u_offset=0.3:  norm=0.070535, residual=6.86e-17
  u_offset=0.45: norm=0.070536, residual=1.86e-16
  Always norm=0.070535-0.070536, always mean=0.000. This is a genuine solution branch.
RESPONSE agent1 to agent2: You also found this branch at phase=pi/2 (line 48: "norm=0.071!") but labeled it TRIVIAL. It's NOT trivial — trivial has norm=0.000000 exactly. This is a distinct 4th branch: small-amplitude sin(θ)-like oscillation, likely the K(θ)-modulated bifurcation from the trivial branch.
NOTE agent1: PHYSICAL INTERPRETATION — The 4th branch likely bifurcates from u≡0 when K_amplitude exceeds a critical threshold. For u''=(u³-(1+K)u), linearizing around u=0 gives u''+u=-Ku. With K=0.3cos(θ), this has a resonant forcing at the natural frequency. The 0.071 norm is the small-amplitude nonlinear response. This is a pitchfork bifurcation branch.
