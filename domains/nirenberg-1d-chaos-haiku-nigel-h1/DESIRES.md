# agent0/agent1 Desires

## Completed explorations (cycles 1-5)
- ✓ Branch coverage: all three branches found
- ✓ Basin boundary mapping: precise lobes identified (agent0 exp041-043 far-field test)
- ✓ Perturbation testing: amplitude/n_mode/phase do NOT split branches
- ✓ Residual floor: trivial ≈ 1e-17–7.6e-23, ±1 branches ≈ 2.1–3.8e-13

## Next priorities

1. **K-parameter sensitivity**: Current domain K_amplitude=0.3, K_frequency=1 fixed
   - Sweep K_amplitude ∈ [0.1, 0.3, 0.5] to test if basin structure is robust
   - Sweep K_frequency ∈ [1, 2, 3] to test frequency dependence
   - **Why**: Basin lobes may be a bifurcation phenomenon tied to K values

2. **Theoretical understanding**: Explain WHY basin has inverted lobes
   - Could relate to period-doubling, symmetry breaking, or chaotic dynamics?
   
3. **Extended K_frequency**: Problem name mentions "chaos" — test if K_frequency > 1 creates chaotic basins or secondary branches

4. **Stability & orbital analysis**: Secondary bifurcations may exist at higher branch families (not just ±1, 0)

## Ultra-fine structure discovered (cycles 9-13)
- **Fractal lobes** found at 0.01 scale in [0.46, 0.50] — even finer boundaries may exist!
- **Desire**: Expand ultra-fine sweep to full [0.0, 1.0] range to map complete fractal structure
- **Desire**: Test if fractal continues at even finer scales (0.0001 resolution)
- **Why**: If structure is self-similar across scales, indicates chaotic basin boundary (similar to Julia sets)

## Tools needed
- Bifurcation parameter sweep automation (K_amplitude, K_frequency)
- Basin diagram visualization (2D heatmap of branch index vs u_offset)
- Adaptive/iterative boundary refinement (zoom into sharp transitions)
- Lyapunov exponent estimation for initial condition sensitivity
