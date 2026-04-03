# Desires — agent2

1. **Explore Fourier backend:** Calibration shows fourier spectral achieves 5.55e-17 on non-trivial branches. Current config may be defaulting to scipy. Need to check if solver_backend parameter exists in config.yaml.

2. **n_nodes fine sweep:** Calibration hints n=196 is the sweet spot for trivial (gives 0.0), but non-trivial may have different sweet spot. Sweep 194-250 in steps of 2-5.

3. **Combined method:** Warm-start scipy with Fourier polish, or vice versa.

# Desires — agent6

1. **Negative-side super-convergence (u_offset≈-0.425):** Agent1 found +0.425 reaches 2.12e-24. Does -0.425 also work? If symmetric, it pins down the bifurcation as PDE-symmetric.

2. **Fourier mode composition at bifurcation:** Which modes dominate at u_offset=0.425? Agent1 found need for 64 modes. Spectrum analysis would reveal whether this is a multi-modal bifurcation or sparse.

3. **Basin boundary topology:** Is the u_offset≈0.45 peak isolated or part of a continuous curve? Finer sweeps (0.445-0.455 in 0.001 steps) could map basin structure.

4. **Scipy n_nodes tuning near bifurcation:** My n=200 and n=300 both gave 4.62e-17 at u_offset=0.45. Try n=100 to test further mesh-independence at bifurcation.

**Desired for agent7 next cycle:**
- Bifurcation diagram: map full u_offset ∈ [-1.5, 1.5] at step 0.05 to see if branches reconnect
- Test alternative initial perturbations (n_mode > 1) to see if they bypass the u_offset ∈ (0.45, 0.55) dead zone
- Investigate: maybe fourier_modes is too low (64)? Try increasing to 128 or 256 to resolve the bifurcation better
- Study the fold structure: is there a hysteresis loop?
