# DESIRES.md

## Context: Solution Space Optimization

We've reached 1.46e-12 on non-trivial branches with n_nodes=392. The following would accelerate further exploration:

## Wished-for Tools/Capabilities

1. **Automatic hyperparameter sweep with early stopping**
   - Current: Manual edits to config.yaml for each n_nodes value
   - Desired: Parameterized run.sh that accepts (u_offset, n_nodes, solver_tol) and outputs residual
   - Benefit: Could map the full n_nodes×tol parameter space more efficiently
   - Estimated speedup: 3–5× (batch-test 50 configurations instead of 10)

2. **Solution visualization/profiling**
   - Current: Only residual and solution_mean/norm outputs
   - Desired: Access to the actual solution u(θ) profiles, perhaps as plots or TSV vectors
   - Benefit: Understand whether the local optimum at n_nodes=392 comes from super-convergence or oscillatory artifacts in higher modes
   - Impact: May reveal that 392 is not actually optimal, just locally minimal

3. **Solver internals access (Newton iteration count, Jacobian conditioning)**
   - Current: Black-box solver with only final residual
   - Desired: Diagnostics like (# of Newton steps, condition number of Jacobian, estimated error)
   - Benefit: Understand why n_nodes≥395 suddenly degrades—is it ill-conditioning or mode coupling?
   - Impact: Could guide whether to increase tolerance, change mesh type (Chebyshev vs uniform), or reformulate

4. **Tolerance sweep independent of mesh**
   - Current: solver_tol and n_nodes are both edited together
   - Desired: Fine-grained control to test (n_nodes=392, solver_tol∈[1e-10, 1e-13]) independently
   - Benefit: Locate the true numerical stability boundary
   - Risk: May crash at tol=1e-12, but now we know to expect it only on ±1 branches

## Lower-Priority Interests

- **Bifurcation analysis**: Agent1 found bistability at u_offset≈0.55. Would like to map the full bifurcation diagram (u_offset ↦ branch & residual).
- **Alternative solvers**: Current solver is Fourier-Newton. Would be interesting to try finite-element BVP solvers (scipy.integrate.solve_bvp) to check if the 1.46e-12 is fundamental to this problem or solver-specific.

## Progress Update (exp069–130)

**Achieved**: 
- Trivial branch optimized to machine precision (1.1e-21)
- Non-trivial branches optimized to 1.46e-12 via n_nodes=392 (2.2× improvement over initial 3.25e-12)
- Identified local optimum at n_nodes=392 with convergence wall at n_nodes≥395

**Still desired** (in priority order):
1. **Bifurcation analysis** (Agent1's discovery): u_offset≈0.55 shows bistability. Full bifurcation diagram needed to understand branch structure.
2. **Solution profile visualization**: Understand whether n_nodes=392 optimum is real or an artifact. Current output only gives residual; actual u(θ) profiles would clarify.
3. **Tolerance optimization**: Test tol∈[1e-10, 1e-13] at n_nodes=392 to locate the true tolerance floor (n_nodes=392 + tol=1e-12 crashes; other combinations untested).
4. **Alternative solver comparison**: Try scipy.integrate.solve_bvp or other BVP methods to check if 1.46e-12 is solver-specific or fundamental to the problem.


## Session 2026-04-03 additions

- **Positive branch Fourier sweep**: Test modes 1–30 on positive branch to confirm negative asymmetry
- **Even tighter tolerances**: Try tol=1e-13 or 1e-14 to see if mode=1 can go below 5.55e-17
- **Hybrid Fourier+Newton**: Can we use Fourier init (mode=1, low-mode projection) to seed scipy's collocation more effectively?
- **Generalization test**: Does mode=1 Fourier success hold for different K(θ) (e.g., K_frequency=2, K_amplitude=0.5)?
- **Solution profile extraction**: Compute u(θ) from mode=1 Fourier solution; compare to scipy's high-mode approximation to understand the asymmetry

## High-Priority (Unlocked by Fourier Breakthrough)

1. **Optimize positive branch Fourier modes**: modes=51 works on negative branch (1.24e-13) but positive is only 2.45e-13. Sweep modes 40-65 on positive branch to find its peak.

2. **Test tighter Newton tolerance on modes=51**: Current newton_tol=1e-11 locks us at 1.24e-13. Test newton_tol ∈ [1e-12, 1e-13, 1e-14] to see if further refinement is possible without crashes.

3. **Solution profiles at modes=51**: Extract u(θ) profile to verify no spurious oscillations and that the solution is physically reasonable.

4. **Branch asymmetry investigation**: Negative (1.24e-13) vs Positive (2.45e-13). Is this due to Newton solver initialization or fundamental asymmetry? Try different u_offset/amplitude combinations on positive.

5. **Hybrid Fourier→Scipy**: Fourier at modes=51 finds high-precision solution in spectral space. Could we feed that to scipy as initial guess to refine further? Or vice versa?

## Medium-Priority (Remaining Unexplored Axes)

- Modes between 40-50: Fill in the modes=40-50 range to confirm modes=51 is truly global optimum
- Phase/amplitude effects on optimal modes: Does phase=π/2, amp=0.05 remain optimal, or does it vary per mode count?
- Alternative Newton tolerances: Explore tol∈[1e-10, 1e-9] to understand tolerance/modes trade-off

## Lower-Priority (Questions for Future Agents)

- Generalization to other (K_amplitude, K_frequency) parameters: Does modes=51 remain optimal for different K functions?
- Fourier vs. alternative spectral methods: Compare Chebyshev polynomials, Hermite, etc. to Fourier.

## agent6 discoveries: Fourier spectral optimization axis (exp319–421)

**Achieved**: Fourier spectral method now validated as superior to scipy collocation:
- Best result: 1.37e-13 (positive) vs scipy's 1.46e-12 (10.6× improvement)
- Optimal config: method=fourier, fourier_modes=48, amplitude=0.05, newton_tol=1e-12
- All 3 branches explored: trivial=0.0, positive=1.37e-13, negative=1.56e-13

**Wished-for Capabilities** (for future acceleration):

1. **Joint (fourier_modes, amplitude, phase, newton_tol) sweep via parameterized harness**
   - Current: Manual config edits for each combination
   - Desired: bash run.sh with args: `run.sh name desc --modes=48 --amp=0.05 --tol=1e-12`
   - Benefit: Could explore 50+ configurations in parallel, find true global optimum
   - Status: Would require modifying run.sh to parse args and dynamically edit config.yaml

2. **Extended precision arithmetic option**
   - Current: Double precision (machine epsilon ≈ 2.2e-16)
   - Desired: Quadruple or arbitrary-precision Newton solver to reach 1e-14, 1e-16
   - Benefit: Validate whether 1.37e-13 is fundamental limit or just float64 rounding
   - Status: Requires rewriting Fourier solver in mpmath or similar

3. **Solution profile output (u(θ) vectors)**
   - Current: Only residual scalar
   - Desired: Write full u(θ) solutions to disk for analysis (phase space, symmetry, stability)
   - Benefit: Understand whether positive/negative asymmetry is real or solver artifact
   - Status: solve.py computes solutions; just needs save-to-file option

4. **Newton solver diagnostics**
   - Current: No visibility into iteration count or Jacobian condition
   - Desired: Log (newton_iterations, jacobian_cond_number, final_residual_norm) per experiment
   - Benefit: Explain why modes=48 vs 51 vs 64 converge differently
   - Status: solve.py already computes these; just needs logging

5. **Bifurcation map via u_offset sweep**
   - Current: Isolated experiments at u_offset={0, ±0.9}
   - Desired: Automated sweep u_offset ∈ [−1.5, +1.5] at 0.1 intervals with Fourier solver
   - Benefit: Visualize full solution manifold, understand branch asymmetry
   - Status: Low-hanging fruit; requires loop over run.sh calls

