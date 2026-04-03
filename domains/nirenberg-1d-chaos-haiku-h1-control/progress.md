# Agent1 Memory-Based Optimization Progress

## Round 3 Status

**Experiments to date:** 4,572  
**Best residuals achieved:**
- Trivial branch: 0.0 (exact u≡0) to 4.22e-28 (near boundary, u=-1.5)
- Positive branch: 1.86e-16 (u=+1.44, Fourier 4modes)
- Negative branch: 1.86e-16 (u=-1.44, Fourier 4modes)
- **NEW:** 4th branch discovered! norm=0.070536, mean=0.0, energy=0.000099 (phase=π/2 perturbation)

## Key Learnings (Memory System)

### Tolerance Paradox
- Loose tolerance (1e-7 to 1e-9) yields tighter residuals than tight (1e-11 to 1e-14)
- Reason: strict tol hits Newton iteration limits; loose tol achieves natural convergence to machine precision
- **Optimal range: 1e-7 to 1e-9**

### Fourier Mode Trade-off
- Fewer modes → lower residual (counterintuitive!)
- 2 modes: 2.00e-16, 4 modes: 1.86e-16, 8: 2.18e-15, 64: 5.65e-13
- Reason: interpolation to 500-pt fine grid introduces roundoff proportional to N

### Basin Structure
- Negative basin: u_offset ≤ -0.56 (sharp boundary)
- Trivial basin: -0.55 ≤ u_offset ≤ 0.55 (wide, stable)
- Positive basin: u_offset ≥ 0.56 (fractal fine structure)
- **Super-convergence zone: u≈0.463 → trivial at 2.2e-17 (Scipy)**
- **Fourier boundary: u≈±0.462-0.463 creates residuals 1e-25 to 1e-28**

### Optimal Parameters (Per Branch)
- **Non-trivial (±1 branches):** n_nodes=300, solver_tol=1e-9, amplitude=0.2-0.5, n_mode=2-4, u_offset=±1.44
- **Trivial:** u_offset=0.0 or near boundary (±0.463), varies by method

## Round 3 Objectives

1. **Systematic 4th branch characterization** — norm=0.070536 is tantalizing; explore phase/amplitude space
2. **Residual competition:** Can we push trivial branch below 1e-28 or non-trivial below 1.86e-16?
3. **Phase landscape:** Phase=π/2 found 4th branch; sweep full [0, 2π) with fine grid
4. **Amplitude coupling:** How does amplitude interact with phase for exotic solutions?
5. **Hybrid methods:** Can Fourier + scipy switching improve certain regions?

## Memory System Design (Agent1-specific)

- Track ideas with residual goals and success probability ranking
- After each experiment, re-rank based on observed results
- Maintain two lists: **proven_configs** (replicate with tweaks) and **frontier_ideas** (new hypotheses)
- Append findings to next_ideas.md with timestamp and branch
