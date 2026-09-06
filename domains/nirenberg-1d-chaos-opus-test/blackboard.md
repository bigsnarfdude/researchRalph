# Blackboard — nirenberg-1d-chaos-opus-test

CLAIM agent0: residual=7.58e-24 mean=0.000 norm=0.000 (evidence: exp003) — branch=trivial. Best config replicated from haiku runs.
CLAIM agent0: residual=3.25e-12 mean=-1.000 norm=1.003 (evidence: exp007) — branch=negative, scipy baseline
CLAIM agent0: **BREAKTHROUGH** residual=9.33e-14 mean=-1.000 norm=1.001 (evidence: exp032) — branch=negative via FOURIER SOLVER. 35x better than scipy floor!
CLAIM agent0: Fourier solver discovery: method=fourier breaks 3.25e-12 ±1 branch floor. 8 modes optimal (less aliasing from cubic). 64→32→16→8 modes: 2.98e-13→1.03e-13→9.84e-14→9.33e-14.
CLAIM agent0: 128 modes WORSE than 64 (1.6e-12 vs 2.98e-13) — cubic aliasing without dealiasing.
CLAIMED agent0: Testing Fourier solver on positive branch
CLAIMED agent1: Positive branch baseline — u_offset=0.9, targeting solution_mean≈+1
CLAIM agent1: residual=3.25e-12 mean=1.000 norm=1.003 (evidence: exp004) — branch=positive. u_offset=0.9, n_nodes=300, tol=1e-11. Excellent convergence.
CLAIMED agent1: Negative branch — u_offset=-0.9, targeting solution_mean≈-1
CLAIM agent1: residual=3.25e-12 mean=-1.000 norm=1.003 (evidence: exp005) — branch=negative. u_offset=-0.9, n_nodes=300, tol=1e-11. Excellent convergence — NO plateau observed, contradicts hypothesis that negative branch is harder.
CLAIM agent1: residual=4.01e-16 mean=0.000 norm=0.000 (evidence: exp006) — branch=trivial. u_offset=0.0, n_nodes=300, tol=1e-11. Near-machine-precision.
CLAIM agent1: ALL THREE BRANCHES MAPPED. Positive (exp004), Negative (exp005), Trivial (exp006). Negative branch converges identically to positive — both at 3.25e-12.
CLAIMED agent1: Basin boundary exploration — finding where u_offset transitions between branches
CLAIM agent1: BASIN STRUCTURE DISCOVERY — with amp=0.28, mode=2, phase=0:
  u_offset 0.0–0.51  → trivial (mean≈0)
  u_offset 0.515–0.55 → NEGATIVE branch (mean≈-1) despite positive offset!
  u_offset 0.57       → trivial (mean≈0) — islands!
  u_offset 0.58       → negative (mean≈-1)
  u_offset 0.59–0.9+  → positive (mean≈+1)
  Evidence: exp008,010,012,014,016,018,019,020,022,024
  The negative branch has a basin that intrudes into positive u_offset space.
  This is NOT a simple three-way partition — there are interleaved islands near u_offset≈0.5-0.6.
CLAIMED agent1: Testing if amplitude/mode affect basin boundaries
CLAIM agent1: BASIN ANTI-SYMMETRY DISCOVERY (amp=0.28, mode=2, phase=0):
  Positive side: 0→0.51=trivial, 0.515→0.58=NEGATIVE, 0.59+→positive
  Negative side: 0→-0.55=trivial, -0.6=POSITIVE, -0.7+→negative
  Evidence: exp027,029,033,035,036,038 (+ prior basin search)
  KEY FINDING: Just beyond the trivial basin, the solver finds the OPPOSITE-sign branch.
  This is NOT a symmetric partition — there are cross-basin intrusions.
  With pure DC (amp=0), u_offset=0.55 still gives negative → intrusion is a property of the BVP, not the perturbation.
  With mode=1 (matching K), u_offset=0.55 gives trivial → mode choice shifts basin boundaries.
  Mode-1 at u_offset=0.6 CRASHES (mesh exceeded) → boundary is numerically sensitive.
CLAIMED agent1: Testing phase dependence of basin structure at u_offset=0.55
CLAIM agent1: POTENTIAL 4TH BRANCH DISCOVERED — norm=0.071, mean=0.000
  Accessible at u_offset=0.55 with phase=pi/2 (mode 2 or 3)
  NOT the trivial u≡0 branch (which has norm=0.000)
  Residual ~5-10e-12 (well-converged, genuine solution)
  Robust to amplitude changes (amp=0.28 and 0.5 give same norm=0.071)
  Robust to mode changes (mode=2 and mode=3 give same norm=0.071)
  Evidence: exp043, exp049, exp051
  This is a small-amplitude oscillatory solution around zero, distinct from u≡0.
  Phase=pi/2 with mode-2 initial guess is key — phase=0 gives negative branch, phase=pi gives trivial.
CLAIM agent1: Phase dependence at u_offset=0.55, mode=2:
  phase=0 → negative branch (mean=-1)
  phase=pi/2 → 4TH BRANCH (norm=0.071, mean=0)
  phase=pi → trivial (norm=0)
  Evidence: exp014 (phase=0), exp043 (phase=pi/2), exp045 (phase=pi)
CLAIMED agent1: Characterizing the 4th branch — energy, wider parameter survey
