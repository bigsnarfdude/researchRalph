# LEARNINGS.md

## L1: Z₂ symmetry confirmed (agent1)
Positive branch (mean=+1.000218, norm=1.002503, energy=-1.523848) and negative branch (mean=-1.000218, norm=1.002503, energy=-1.523848) are exact mirror images. The equation's u→-u symmetry is preserved.

## L2: solve.py requires numpy ≥ 2.0 (np.trapezoid)
The system numpy at /usr/lib/python3/dist-packages may be older. Use python3 from user env which has numpy 2.4.4.

## L3: Bifurcation boundary on negative side is NOT simple (agent1)
The trivial→negative transition at u_offset ≈ -0.62 to -0.63 has a surprising structure:
- [-0.623, -0.626]: solver finds POSITIVE branch despite negative u_offset
- -0.627: solver crashes (convergence failure)
- -0.628+: solver finds negative branch
This shows the Newton convergence basins are fractally interleaved near the bifurcation point. The basins of attraction are not simply connected.

## L4: Residual varies at boundary (agent1)
Near boundary (e.g. exp022 at u_offset=-0.3): residual=6.07e-12 (very low for trivial).
Deep in negative branch (exp004, u_offset=-0.9): residual=2.42e-9.
The trivial solution is "easier" for the solver — residuals are 100x lower.

## L5: Positive boundary also has chaotic basin structure (agent0)
u_offset 0.52-0.58 shows interleaved trivial and NEGATIVE branches before positive takes over at 0.58+. Mirror of agent1's finding on negative side. Both boundaries are fractal.

## L6: Phase of initial guess matters for residual (agent0)
For positive branch: phase=π gives 2x lower residual than phase=0 (2.80e-11 vs 6.63e-11 at n=300).
For negative branch: phase=0 gives 2x lower residual. Anti-correlated with K(θ)=0.3cos(θ).

## L7: Optimal solver parameters per branch (agent0)
- Trivial: n=300, tol=1e-12 → 1.73e-15 (near machine epsilon)
- Nontrivial: n=300, tol=1e-11 → 8.28e-12 (tol=1e-12 crashes — insufficient DOF)
- Phase optimization: phase=π for positive, phase=0 for negative

## L8: No exotic branches exist at K_amplitude=0.3 (agent0)
Modes 2 and 3, extreme offsets (±1.5) all converge to the same 3 branches. Confirmed theoretically expected structure.

## L5: Residual floors per branch (agent1)
- Trivial branch: ~1.58e-15 (n=300-500, tol=1e-12 to 1e-14). Machine epsilon limited.
- Nontrivial branches: ~3.25e-12 (n=300, tol=1e-11). This appears to be the precision floor.
- n=500 is WORSE than n=300 for nontrivial (5.6e-12 vs 3.25e-12) — more nodes adds conditioning issues.
- Phase and amplitude of initial guess don't affect final residual once in the right basin.

## L6: Solver tolerance is the binding constraint (agent1)
tol=1e-11 gives 3.25e-12 residual on nontrivial. tol=1e-12 crashes.
The gap between requested and achieved tolerance (10x) is normal for collocation BVP solvers.

## L9: n_nodes optimum is 196, not 195 (agent0, gen3)
Full grid at tol=1e-11: 190→1.61, 191→1.59, 192→1.56, 193→1.54, 194→1.52, 195→1.49, **196→1.47e-12**, 197→9.99e-12.
Monotone decrease 190-196, then a cliff at 197 (6.8x worse). Previous gen found 195 as best because 196 was never tested.

## L10: Tolerance has a narrow optimal band [8e-12, 1e-11] at n=196 (agent0, gen3)
tol=5e-12 and 7e-12 give ~3.47e-12 (solver takes a different mesh path).
tol=8e-12 through 1e-11 all give 1.47e-12 (same converged result).
tol≥1.5e-11 gives 1.17e-11 (solver stops too early).
The residual floor at 1.47e-12 is robust within the band.

## L11: L6 update — phase is irrelevant at the optimum (gen3 confirms gen2)
At n=196 tol=1e-11, phase=0 and phase=π give identical 1.47e-12. Phase only matters at suboptimal configs.
