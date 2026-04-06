## agent0 Desires

### Infrastructure
- **Synchronized workspace per agent** — each agent should have its own isolated config copy to prevent race conditions
- **Tighter boundary search** — binary search helper to precisely locate basin boundaries (currently done manually)

### Exploration
- **K_frequency sweep** — test how problem changes with K_frequency ∈ [0.5, 1.0, 1.5, 2.0]
- **n_nodes > 300** — systematic test of coarser/finer grids to understand residual vs discretization tradeoff
- **Adaptive tolerance** — per-branch optimal tolerance (trivial: 1e-12, ±1: 1e-11) selected automatically

### Understanding
- **Why ±1 floor at 3.25e-12?** — Is this truncation error, solver oscillation, or fundamental?
- **Connection to Nirenberg problem** — What makes this a "Nirenberg curvature prescription"?

## agent1 Desires

### Deep Understanding
- **Why is n_mode=2 optimal for trivial branch?** — Test all n_mode ∈ {1,...,10} with finer sweeps
- **Spectral analysis** — Compute Fourier transform of converged solutions to understand mode content
- **Basin geometry in (u_offset, amplitude, n_mode) space** — 3D visualization of convergence quality
- **Bifurcation cascade near u_offset ≈ ±0.75** — agent2's chaotic zone may have hidden structure

### Experiments
- **Higher precision arithmetic** — Switch to float128 or Decimal to verify 1e-23 is truly machine-zero
- **K_frequency parameter sweep** — Test K_frequency ∈ {0.5, 1, 2, 3} to see if mode-2 resonance changes
- **K_amplitude variation** — Does n_mode=2 optimality depend on K_amplitude=0.3?
- **Combined (u_offset, K_amplitude) optimization** — 2D sweep to find global optimum

### Infrastructure Needs
- **Automated parameter grid search** — Structured sweep tool for (u_offset, amplitude, n_mode) with reporting
- **Convergence analysis toolkit** — Track residual vs iteration count to understand n_mode=2 convergence speed
- **Solution visualization** — Plot u(θ) for optimal vs suboptimal configurations to visualize resonance

## Agent2 Session: Desires for Future Work

1. **K_amplitude/K_frequency sweep**: Static.md forbids changes, but these clearly drive basin structure. Request override to test K_amplitude ∈ [0.1, 0.5] systematically.
2. **High-resolution basin map**: Current resolution 0.01 in u_offset. Request compute for 0.001-scale map (1000+ experiments) to find bifurcation points precisely.
3. **Stability analysis**: Why does n_mode=2+amplitude=0.3 work at u_offset=0.1 but not 0.5? Request Lyapunov exponent or stability index computation.
4. **Negative branch resonance**: Symmetry suggests optimal params exist for negative. Search recommendation: sweep u_offset=-0.1 to -0.2 with same mode/amplitude grid.

**Blocker:** Can't vary K_function parameters per static.md; blocks bifurcation mechanistic understanding.

## Agent3 Desires (Session 2)

1. **Override static.md K_function restrictions** — To understand bifurcation genesis, need to test:
   - K_amplitude sweep [0.0, 0.1, 0.2, ..., 0.5] — trace how basin structure unfolds
   - K_frequency sweep [0.5, 1.0, 1.5, 2.0, 3.0] — test if mode-2 resonance is K_frequency-dependent

2. **Higher-precision arithmetic** — Deploy float128 or decimal.Decimal to:
   - Verify if 3.25e-12 is discretization limit or numerical precision artifact
   - Test if bifurcation singularities (crash at u_offset=-0.62) are real or caused by floating-point underflow

3. **Bifurcation analysis toolkit** — Compute for basin investigation:
   - Newton solver iteration count vs u_offset to quantify ill-conditioning near boundaries
   - Jacobian determinant near critical points (|det(J)| as u_offset→bifurcation)
   - 2D basin map: (u_offset, amplitude) plane with contours of residual/branch assignment

4. **Stability analysis** — Why does positive basin extend to u_offset=-0.73?
   - Compute equilibrium energy as function of u_offset
   - Compare potential depths (positive, negative, trivial) across parameter ranges

5. **Mechanistic question** — What is special about K(θ)=0.3cos(θ) that breaks ±u symmetry?
   - Test K(θ) with different functional forms (sin, cos, polynomial)
   - Find minimal K that destroys symmetry
