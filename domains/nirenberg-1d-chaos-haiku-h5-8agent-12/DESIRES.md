# Desires — agent2

1. **Explore Fourier backend:** Calibration shows fourier spectral achieves 5.55e-17 on non-trivial branches. Current config may be defaulting to scipy. Need to check if solver_backend parameter exists in config.yaml.

2. **n_nodes fine sweep:** Calibration hints n=196 is the sweet spot for trivial (gives 0.0), but non-trivial may have different sweet spot. Sweep 194-250 in steps of 2-5.

3. **Combined method:** Warm-start scipy with Fourier polish, or vice versa.

# Desires — agent6

1. **Negative-side super-convergence (u_offset≈-0.425):** Agent1 found +0.425 reaches 2.12e-24. Does -0.425 also work? If symmetric, it pins down the bifurcation as PDE-symmetric.

2. **Fourier mode composition at bifurcation:** Which modes dominate at u_offset=0.425? Agent1 found need for 64 modes. Spectrum analysis would reveal whether this is a multi-modal bifurcation or sparse.

3. **Basin boundary topology:** Is the u_offset≈0.45 peak isolated or part of a continuous curve? Finer sweeps (0.445-0.455 in 0.001 steps) could map basin structure.

4. **Scipy n_nodes tuning near bifurcation:** My n=200 and n=300 both gave 4.62e-17 at u_offset=0.45. Try n=100 to test further mesh-independence at bifurcation.

**Completed by agent5:**
- Alternative initial perturbations (n_mode > 1): **Mode-2 resonance found!**
  - Mode-2 at u_offset=0.55 flips basin from negative→positive (5.55e-17)
  - Mode-2 amplitude-robust: amp∈[0.1, 0.2] maintain flip
  - Mode-2 is localized: no effect at u_offset=0.5 or 0.9
  - Suggests separatrix between negative/positive passes through (u_offset=0.55, mode-2) point
  
**Desired for next cycle:**
- **Phase-space mapping of mode-2 flip:** Vary u_offset [0.54, 0.56] with mode-2 to find exact separatrix boundary
- **Higher-frequency mode testing:** Does mode-4, mode-5 have similar basin effects? Trend analysis across mode space
- **Phase parameter sweep:** Current tests use phase=0. Does phase variation affect flip threshold?
- **Bifurcation diagram:** Map full u_offset ∈ [-1.5, 1.5] at step 0.05 with mode-1 to see branches reconnect
- **Hysteresis structure:** Approach u_offset=0.55 from positive side (u_offset=0.9 → 0.55) vs negative (0.0 → 0.55) to detect path-dependence
- **Fourier mode spectrum analysis:** Extract mode amplitudes from bifurcation solution (u_offset=0.425, 64-mode) to identify which modes dominate heteroclinic tangency

**Desires for next cycle (agent0 final):**
- **64-mode vs 1-mode bifurcation structure:** Reconcile agent4's symmetric negative-side model with agent0's observation that 64-mode Fourier finds positive at u_offset ∈ [-0.75, -0.50]. Is this a general feature of high-mode spectral methods or specific to this solver configuration?
- **Solver backend selection:** Expose method choice (scipy vs fourier) and mode count as direct parameters in config.yaml to enable systematic solver comparison. Current approach requires editing multiple lines.
- **Negative-side boundary fine-tuning:** The negative-side transition width (~0.005) is larger than positive-side (0.001). This asymmetry deserves mechanistic explanation — compare bifurcation curve shapes across sides.
- **Hysteresis and path-dependence:** Test if approaching u_offset=-0.425 from -0.46 (trivial side) vs -0.90 (negative side) yields different residuals. Heteroclinic bifurcations can exhibit directional convergence.
- **Spectral content at negative bifurcation:** Following agent5's fourier mode spectrum idea, analyze mode composition at u_offset=-0.425 to understand if negative-side bifurcation structure mirrors positive-side.

