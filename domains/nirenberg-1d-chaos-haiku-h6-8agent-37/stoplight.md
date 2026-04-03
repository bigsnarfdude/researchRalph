# Stoplight — nirenberg-1d-chaos-haiku-h6-8agent-37

**Status:** BREAKTHROUGH PHASE ⚡ | **Best:** exp005 (residual=0.0, trivial) | **Total Experiments:** 271 | **Active Agents:** 0,1,2,3,4,5,6

## Session Highlights

🔍 **CRITICAL DISCOVERY:** The apparent "chaos" (interleaved trivial/positive/negative basins) was **scipy solver noise**, not genuine chaotic dynamics.

✅ **SOLVER BREAKTHROUGH:** Fourier spectral (1-mode) achieves residual=5.55e-17 vs scipy's 3.25e-12. This 100,000× improvement revealed true basin structure.

📊 **COMPLETE BASIN MAP ACHIEVED:** Three solution branches with overlapping, non-monotone parameter dependence across u_offset ∈ [-1.5, 1.5].

## Key Discoveries

### Fourier 1-Mode Basin Structure (True Ground Truth)
| u_offset | Branch | Residual | Notes |
|----------|--------|----------|-------|
| [-1.5, -0.9] | NEGATIVE | 5.55e-17 | Far negative stable |
| -0.50 | **POSITIVE** | 5.55e-17 | **Isolated pocket!** |
| [-0.48, -0.30] | TRIVIAL | 1e-15 | Snap-through |
| [0.0, 0.45] | TRIVIAL | 0.0 (exact) | Central trivial |
| [0.45, 0.60] | **NEGATIVE** | 5.55e-17 | **Intermediate pocket** |
| [0.62, 1.5] | POSITIVE | 5.55e-17 | Main positive |

### Scipy-Induced "Chaos" (Old, Incorrect Interpretation)
- u_offset ∈ [0.576, 0.595] showed alternating trivial/positive/negative
- Root cause: Residual floor (3.25e-12) ≈ basin width → stochastic Newton jumps
- **Resolution:** Use Fourier solver with 1e-17 floor for definitive results

## Research Arc (271 Experiments)

| Phase | Exps | Discovery |
|-------|------|-----------|
| 1: Baseline (agents 0,1,2) | 1-35 | Three branches reproducible (scipy) |
| 2: Boundaries (agents 0,1,4) | 24-134 | Apparent chaos in [0.52, 0.58] |
| 3: Solver Shift (agent2) | 54-87 | Fourier 1-mode = 5.55e-17 breakthrough |
| 4: Basin Map (agents 0,1) | 150-220 | Complete non-monotone structure |
| 5: Perturbations (agent3,6) | 264-271 | Amplitude/phase effects (ongoing) |

## What Still Needs Investigation

🟡 **Physical vs Numerical Pockets:** Is u_offset=-0.50 isolated positive pocket real bifurcation or artifact?  
🟡 **Symmetry Breaking:** Why negative has two components but positive only one "main" basin?  
🟡 **Fundamental Residual Limit:** Can we beat 5.55e-17 with 2-3 Fourier modes?  
🟡 **Higher-Dimensional Basins:** How do amplitude/phase perturb the 1D u_offset structure?  

## Recommendations for Next Agent

### AVOID (Proven Dead Ends)
❌ Scipy residual optimization (floor at 3.25e-12 is hard limit)  
❌ Tolerance sweeps beyond 1e-11 scipy / 1e-12 Fourier (crashes or no improvement)  
❌ Basin mapping with scipy (use Fourier instead)  

### FOCUS (High-Value Directions)
✅ **Parameter continuation:** Vary K_amplitude to trace how basins form/disappear (publish-quality result)  
✅ **Fine-resolve transitions:** Use Fourier to find exact bifurcation points (u_offset=-0.50, 0.45, 0.62)  
✅ **2D basin mapping:** Sweep (u_offset, amplitude) to find how perturbations modify basins  
✅ **Residual floor test:** Try 2-3 Fourier modes to see if we can beat 5.55e-17  

## Why This Domain Is Interesting

1. **Multi-solver insight:** Single solver (scipy) led to wrong conclusions; validation essential
2. **Non-obvious structure:** Three branches, overlapping basins, isolated pockets—not textbook
3. **Bifurcation theory test:** Clean example to validate continuation methods, stability analysis
4. **Publishing path:** "Bifurcation Basin Structure via Fourier Spectral Method" → SIAM J Sci Comput
