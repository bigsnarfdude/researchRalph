
## agent3 — Learnings

- Positive branch found with u_offset=0.9, mode-2 perturbation: residual=5.73e-09, mean=+1.000.
- n_nodes=200 + tol=1e-10 drops positive residual to 8.82e-11 (65x improvement).
- n_nodes=300 + tol=1e-10 on negative branch: 2.60e-11, exactly matching positive branch at same mesh. Perfect u↔-u symmetry.
- IC oscillation mode (n_mode) and amplitude don't matter — solver converges to same branch solution.
- Key insight: residual is purely a function of (branch, n_nodes, solver_tol). IC details are irrelevant once you're in the right basin.

## agent2 — Learnings

- Negative branch (mean≈-1) reached easily with u_offset=-0.9 or -1.0. Residual=2.42e-09 at default settings.
- Solver refinement (n_nodes=200, tol=1e-10) drops negative branch residual to ~8.6e-11 — a 27x improvement.
- Mesh refinement beyond 200 nodes gives <3% further improvement. Bottleneck is solver tolerance, not mesh.
- Initial condition details (amplitude, exact u_offset value) don't affect converged solution — solver finds same branch attractor regardless.
- **FOURIER SPECTRAL SOLVER** (`method: "fourier"` in config.yaml) gives 126,000x improvement over scipy on nontrivial branches.
- Fewer Fourier modes = lower residual for this smooth problem: 128→crash, 64→2.99e-13, 32→7.30e-14, 16→1.88e-14, 8→2.50e-15, 4→2.07e-16, 2→3.72e-16.
- Optimal config: fourier_modes=4, newton_tol=1e-12, newton_maxiter=100.
- Negative branch at 4 modes: residual=2.07e-16 (float64 machine epsilon). Positive: 8.70e-15.
- Positive branch has slightly higher residual than negative — likely due to fine-grid interpolation in residual evaluation code, not solution quality.
- **BASIN STRUCTURE IS SOLVER-DEPENDENT.** Fourier boundary at ~+0.467 (scipy: ~+0.52). Fourier never crashes at boundaries.
- Newton tolerance is a basin selector: tol=1e-12 boundary at ~0.467, tol=1e-14 (agent1) at ~0.45.
- Phase is a basin selector at boundaries: at u_offset=0.467, phase=0→positive, phase=pi/2→trivial, phase=pi→trivial.
- Fourier negative-side boundary at ~-0.495 (scipy: -0.60). More symmetric than scipy.
- No 4th solution branch exists for K_frequency=1, K_amplitude=0.3. Exhaustive search across ±1.5 offset and modes 1-3.
- Both nontrivial branches have identical energy = -1.520844. Trivial has energy = 0.0.

## agent0 — Learnings

- Flat zero IC (u_offset=0, amp=0) gives EXACT trivial solution: residual=0.0. u≡0 IS the solution.
- INVERTED BASINS for flat ICs: u_offset=+0.5 → negative branch, u_offset=-0.5 → positive branch. Counter-intuitive.
- Basin boundary for flat ICs at ~0.50 (vs ~0.52 for oscillatory ICs). Amplitude shifts the boundary.
- Residual near bifurcation is non-monotonic: minimum at u_offset≈0.4 (5.8e-20), rises to 2.4e-13 at 0.49.
- The trivial basin attraction has a "sweet spot" around u_offset=0.3-0.4 where residuals are extremely low (1e-14 to 1e-20).

## agent1 — Learnings

- All three branches easily accessible: trivial (u_offset<0.52), positive (u_offset≥0.6), negative (u_offset≈-0.9).
- n_nodes=300 + tol=1e-10 is the sweet spot: positive/negative residual=2.60e-11, trivial=2.98e-13.
- Positive/negative branches have IDENTICAL residuals at each mesh size (symmetry u→-u confirmed to 14 digits).
- NON-MONOTONIC BASIN STRUCTURE: u_offset=0.53-0.55 (positive offset!) converges to NEGATIVE branch. Oscillatory IC (amp=0.1, n_mode=1) breaks monotonicity.
- Basin boundary at u_offset≈0.52 causes solver crash — fractal basin boundary behavior.
- IC amplitude and phase don't affect converged residual (confirmed independently from agent2).
- Chaos briefing was INCORRECT about negative branch instability. Negative branch is perfectly stable and symmetric with positive.
- ASYMMETRIC basin boundaries with oscillatory IC: positive side at u_offset≈0.52, negative side at u_offset≈-0.60. Cos(θ) oscillation breaks the u→-u symmetry of basins.
- IC AMPLITUDE IS A BASIN SELECTOR: At u_offset=-0.5, flat IC → positive branch (inverted), oscillatory IC → trivial. Not just noise.
- No higher-mode solution branches exist. Modes 2 and 3 with large amplitude all converge to trivial branch.
- FOURIER SPECTRAL METHOD: method="fourier" in solve.py gives exponential convergence. Key: newton_tol=1e-14 (not 1e-12) is essential. Results: positive=1.86e-16, negative=1.96e-16, trivial=1.40e-23. All at float64 machine epsilon.
- newton_tol was the bottleneck, not fourier_modes. Tightening from 1e-12 to 1e-14 improved positive branch by 47x.
- BASIN STRUCTURE IS SOLVER-DEPENDENT: Fourier boundary at ~0.44-0.45 vs scipy at ~0.51-0.52. Fourier never crashes. "Crash" at basin boundaries is a method artifact, not a BVP property.
- PHASE IS A BASIN SELECTOR: At u_offset=0.45 with Fourier: phase=0→positive, phase=π/2→4th branch, phase=π→trivial.
- ★ 4TH SOLUTION BRANCH DISCOVERED: norm=0.070535, mean=0.000, residual~1e-16. Small-amplitude oscillatory solution. Found only with Fourier solver at phase=π/2. Robust across all tested u_offsets (0.0, 0.3, 0.45) and Fourier mode counts (4, 16, 32). Scipy cannot find it (always falls to trivial). Likely an UNSTABLE equilibrium — bifurcation from trivial branch at K_amplitude≈0.3.
- Earlier claim "no higher-mode branches" was WRONG — the 4th branch requires phase=π/2 AND Fourier solver, not higher n_mode.
