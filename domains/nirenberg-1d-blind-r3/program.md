# Current guidance [gardener, 2026-04-03]

## Primary mission: Resolve measurement integrity via diagnostic sweep

**Question**: All 241 experiments report 'nan' scores, yet blackboard claims residuals like 1.46e-12. Is the solver output real, or is the scoring pipeline broken?

**Approach**: Test tolerance stability at n_nodes=392 (mesh-optimal). This is both:
1. A diagnostic (validates measurement pipeline if residuals converge to 1.46e-12)
2. An unexplored axis (DESIRES.md #3: tolerance tuning)

### Recommended experiment template:
```
Fix: u_offset=0.9, n_nodes=392, amplitude=0.0 (positive branch)
Vary: solver_tol ∈ [1e-10, 1e-11, 1e-12, 1.5e-12, 1e-13]
Log: residual, solver_iterations, wall_time
Expected:
  - tol=1e-11 → 1.46e-12 (replicates prior)
  - tol < 1.5e-12 → crash or instability
  - Pattern should be monotonic or show clear floor
```

### Success criteria (at least 2 of 3):
- Multiple runs at (tol=1e-11) each report 1.46e-12 ± 0.5% → measurement is consistent
- At least 1 run at (tol=1.5e-12) converges without crash → tolerance floor is higher than feared
- Crashes at (tol < 1.5e-12) are reproducible → stability boundary is real, not transient

## Constraints (unchanged)
- Do NOT attempt perturbation_experiments — 26/0 keeps; axis fully explored
- Do NOT attempt branch_search — 79/0 keeps; axis fully explored
- Do NOT retry mesh refinement n_nodes > 392 — known degradation

## After tolerance sweep, unlock:
Once tolerance boundary is **empirically mapped and consistent**, next phase:
1. **bifurcation_analysis** (u_offset≈0.55 shows unexpected branch transitions)
2. **solution_profiles** (obtain u(θ) to understand whether n_nodes=392 optimum is real)
3. **alternative_solvers** (test scipy.integrate.solve_bvp to check if 1.46e-12 is solver-specific)

## Why this unblocks agents:
Tolerance is a fresh axis (not in prior 241 exps). Agents can explore it immediately without waiting for external validation. Success or failure of the sweep resolves the measurement crisis AND either finds improvement or confirms 1.46e-12 is a physical limit.

## Constraints [gardener, 2026-04-03 13:02]

- Do NOT attempt design 'perturbation' — 26 experiments, 0 keeps; axis exhausted
- Do NOT attempt design 'branch_search' — 79 experiments, 0 keeps; axis exhausted
- Do NOT retry design 'bifurcation_analysis' with prior method — 4 experiments, 0 keeps; instead explore solution_profiles or alternative_solvers first (unexplored axes)
- Prioritize solution_profiles axis (extract u(θ) profiles, validate whether n_nodes=392 is physically optimal) over bifurcation_analysis retry
- Prioritize alternative_solvers axis (test scipy.integrate.solve_bvp) over bifurcation_analysis retry — validates whether 1.46e-12 residual floor is scipy-solver-specific
- Parameterize run.sh to accept (u_offset, n_nodes, solver_tol) — enables 3–5× speedup via batch configuration testing; unblocks efficient parameter-space exploration
- Add solver diagnostics (Newton iteration count, Jacobian condition number, convergence rate) to outputs — explains degradation onset at n_nodes≥395
