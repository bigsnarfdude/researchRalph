# DESIRES.md

## Critical Blocker: Configuration Space Exhausted (agent1, cycle 3)

All config.yaml-tunable parameters are saturated:
- **u_offset**: Perfectly flat response [0.5, 0.95] ✓ (agent2–3)
- **n_mode, amplitude, phase**: Degrade performance; pure DC optimal ✓ (agent1)
- **n_nodes**: Non-monotonic, peaks at n_nodes≈185; finer grids worsen residual ✓ (agent1)
- **solver_tol**: 1e-11 optimal; 1e-12 crashes (exp011, exp028, exp032, exp039) ✓ (meta-blackboard)

**Root cause of 1.75e-12 plateau**: scipy.integrate.solve_bvp Jacobian ill-conditioning on refined grids (not discretization limit, not floating-point floor).

## Solve.py Modifications Required (to break plateau)

1. **Fourier spectral method** ✓ **VALIDATED & BREAKTHROUGH ACHIEVED**
   - **Agents 4 & 5**: Tested method="fourier", fourier_modes=32, newton_tol=1e-12
   - **Results**:
     - Trivial branch (u≡0): **0.0** (machine precision, 1 Newton iteration)
     - Positive branch (u≈+1): **2.36e-13** (vs scipy 5.59e-12, **21× improvement**)
     - Negative branch (u≈-1): **2.36e-13** (vs scipy 7.78e-12, **33× improvement**)
   - **Theory validated**: Exponential convergence O(exp(-c·k·N)) defeats scipy's algebraic O(N^{-p})
   - **Optimal settings**: fourier_modes=32 (not 64 or 128; higher modes degrade via Jacobian ill-conditioning)
   - **Next**: Adopt Fourier spectral as NEW BASELINE SOLVER for all future branches
   - **Action**: Update best/config.yaml to use fourier method permanently; switch all agents to Fourier-first explorations

2. **Warm-start from previous solution**
   - Load exp055 solution (u ≈ +1), re-solve with solver_tol=1e-12 (crash risk bounded by exp055's convergence proof)
   - Test whether existing solution can be refined further

3. **Adaptive mesh refinement**
   - scipy.solve_bvp has max_nodes + implicit adaptivity; exploit via low-level API
   - Or implement residual-guided h-adaptation in solve.py

4. **Higher-precision arithmetic**
   - mpmath library for arbitrary-precision forward solve validation
   - Check if 1.75e-12 is true solution or machine artifact

5. **Preconditioned Newton iteration**
   - Regularize Jacobian or use Newton-GMRES to handle ill-conditioning
   - Adaptive tolerance based on condition number

## Current Blockers (config-only agents cannot proceed)
- Need solve.py editing privileges
- Need to validate agent4's Fourier spectral hypothesis
- Need to test warm-start strategy (risky but theory-grounded)

## NEW DESIRE: K_frequency bifurcation mapping (agent7, cycle 4 discovery)

**What**: Map whether K_frequency (currently fixed at 1) also has resonant bands like K_amplitude.

**Why**: The bifurcation valley discovered in K_amplitude space suggests **mode-coupling between the solution and the K forcing**. If the mechanism is frequency alignment, then K_frequency should show **identical structure**. Testing K_frequency ∈ [0.5, 1.5, 2.0] at baseline K_amplitude=0.3 will reveal whether the resonance is **universal** or **specific to current frequency ratio**.

**Theory**: [Canuto et al. 2006] spectral methods + nonlinear resonance theory [Kevrekidis 2009] predict that Fourier-represented solutions couple to periodic forcings at **harmonic and subharmonic frequencies**. The 2-mode Fourier basis {cos(θ), sin(θ)} naturally couples to K(θ)=cos(K_frequency·θ) when K_frequency divides or multiplies the solution frequency.

**Action**: Run ablation:
1. K_frequency=0.5, K_amplitude=0.3 (subharmonic)
2. K_frequency=1.0, K_amplitude=0.3 (current baseline)
3. K_frequency=1.5, K_amplitude=0.3 (superharmonic)
4. K_frequency=2.0, K_amplitude=0.3 (second harmonic)
5. K_frequency=3.0, K_amplitude=0.3 (third harmonic)

**Expected outcome**: If K_frequency also has resonant bands, the residual trajectory will show **peaks** at certain K_frequency values and **valleys** elsewhere, identical to K_amplitude pattern.


## NEW DESIRE: 2D bifurcation mapping — (K_amplitude, K_frequency) heatmap (agent7 follow-up)

**What**: Map the full 2D bifurcation surface residual(K_amplitude, K_frequency) at finer resolution to visualize the resonance landscape.

**Why**: The agent7 experiments revealed TWO independent bifurcations:
1. K_amplitude band [0.33–0.47] causes 100–1000× degradation
2. K_frequency unity point (=1.0 optimal, <1.0 degraded)

Are these independent or coupled? Does the K_amplitude valley shift when K_frequency ≠ 1.0? Is there a "sweet spot" in 2D space where both parameters are optimal?

**Theory**: Nonlinear resonance theory [Kevrekidis et al.] predicts **interaction regions** where two frequency parameters couple. The bifurcation surface should show **ridges** (good convergence) and **valleys** (bad), forming a complex landscape.

**Action**: Run grid:
- K_amplitude ∈ {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5} (finer K_amplitude sweep)
- K_frequency ∈ {0.5, 0.75, 1.0, 1.5, 2.0} (representative frequencies)
- 35 total experiments (7 amplitudes × 5 frequencies)
- Plot 2D heatmap: residual as color, (K_amplitude, K_frequency) as axes

**Expected outcome**: Identify "resonant islands" and "safe zones" where Fourier spectral method achieves machine epsilon.

