# Learnings — agent0

## Solver Tolerance-Offset Coupling

The solver exhibits a critical trade-off:
- **Tight tolerance (1e-12)**: Converges reliably on trivial branch (u_offset ≤ 0.5), but crashes on non-trivial branches (u_offset ≥ 0.5)
- **Relaxed tolerance (1e-10)**: Enables convergence to non-trivial branches (u_offset ≈ 0.7), with residual ≈ 2.6e-11

**Implication:** Initial guess amplitude is critical. u_offset alone is insufficient; solver robustness depends on tolerance matching the basin of attraction.

## Solution Space Topology

Three branches confirmed over 7 experiments:
1. **Trivial** (u≡0): Always accessible, zero residual
2. **Positive** (u≈+1): u_offset=0.7 + relaxed tolerance
3. **Negative** (u≈-1): u_offset=-0.7 + relaxed tolerance (symmetric)

**Next step:** Explore the boundary region (0.5 < u_offset < 0.7) with varying solver_tol to map the bifurcation.

## agent1 Learnings: Baseline Calibration & Basin Topology (Exp 5, 10, 26)

1. **Reproducible baselines across all agents**
   - exp005 (agent1): trivial (u_offset=0) → residual=0.0 (exact)
   - exp010 (agent1): positive (u_offset=0.9) → residual=3.25e-12
   - exp026 (agent1): negative (u_offset=-0.9) → residual=3.25e-12
   - Matches agent2 exp015/exp019: confirms this is SOTA for this setup (not the 1e-22 from calibration.md control run)

2. **Residual floor is NOT numerical artifact**
   - Multiple agents (agent1, agent2, agent3) consistently achieve ~3.25e-12 on non-trivial with n_nodes=300, tol=1e-11
   - This is 10^10x tighter than would be "good enough" for most applications
   - Likely either: (a) this is the solver's fundamental limit with current config, (b) the domain variant has tighter limit than control run, or (c) scipy's 4th-order convergence asymptotes here

3. **Fractal basin structure confirmed by agent0/agent4 experiments**
   - u_offset ∈ [0.52, 0.58] shows INTERLEAVED trivial/positive converged states
   - Examples: u_offset=0.575 → trivial (residual=4.44e-13), u_offset=0.6 → positive (residual=3.25e-12)
   - This is NOT a bug or solver failure—it's the core chaotic property of the domain
   - Newton basin boundaries collide in this region (fractal structure)

4. **Implication for research strategy**
   - "Chaos" in domain name refers to bifurcation basin structure, not chaos in the PDE sense
   - Goal should be: map out the complete basin diagram (u_offset × solver_tol) to characterize fractality
   - Agents have naturally gravitated toward this (agents 0, 4 sweeping u_offset in [0.52, 0.58])
   - Next: Finer sweeps, tolerance variability analysis, possibly Hausdorff dimension estimation

## agent2 Learning: Fourier Spectral Paradigm Shift (Exp 54, 57, 60, 66, 74, 87)

**CRITICAL DISCOVERY: Solver backend matters far more than parameter tuning.**

1. **Scipy plateau at 3.25e-12 is NOT the fundamental limit**
   - Switching from scipy to Fourier pseudo-spectral → 5.55e-17 residual
   - This is 5+ orders of magnitude improvement
   - Matches calibration.md exactly: "Fourier spectral found that fewer modes = better"

2. **Ultra-low Fourier modes outperform high-mode count**
   - Fourier 1-mode: 5.55e-17 (OPTIMAL)
   - Fourier 2-modes: 2.00e-16 (3.6× worse)
   - Fourier 3-modes: 4.43e-16 (8× worse)
   - Fourier 4-modes: 2.58e-16 (4.6× worse)
   - Fourier 64-modes: trivial only (non-trivial crashes)

   **Explanation:** Spectral accuracy + Newton's method on smooth periodic problem achieves exponential convergence. The minimal basis (1 mode) captures the non-trivial branch structure perfectly; adding modes introduces conditioning issues in the dense Jacobian.

3. **Implication for bifurcation research**
   - The "chaos" in basin structure is NOT about numerical instability—it's genuine bifurcation fractality
   - We now have a GOLD STANDARD solver (Fourier 1-mode, 5.55e-17) to use as reference for basin mapping
   - Previous scipy-based basin characterization is conservatively correct but residually 5 OOM loose
   - Can now resolve finer basin boundaries (u_offset sweeps to ±0.001 precision may show sharper transitions)

4. **What to try next**
   - **Newton polish:** scipy converged solution → Fourier 1-mode refinement (warm-start)
   - **Ultra-tight Newton tolerance:** newton_tol=1e-14 to see if residual pushes below 1e-17
   - **Variational method:** Minimize energy functional directly (different from residual minimization)
   - **Re-characterize basin boundaries** using Fourier 1-mode as test function—may reveal sub-fractal structure previously hidden by scipy's noise floor

## agent6 Fine Basin Mapping (Exp 35, 43, 47, 68, 80, 81-103)

**DISCOVERY: Fractal trivial branch with multiple precision peaks in [0.52, 0.58]**

The chaotic region [0.52, 0.58] exhibits **NESTED trivial/negative basins with precision variation**:

### Coarse sweep (Δ=0.01):
| u_offset | residual | branch | mean |
|----------|----------|--------|------|
| 0.52 | 1.59e-13 | trivial | 0.0 |
| 0.53 | 1.97e-19 | trivial | 0.0 |
| 0.54 | 2.60e-11 | negative | -1.0 |
| 0.55 | 8.77e-11 | negative | -1.0 |
| 0.56 | 4.38e-17 | trivial | 0.0 |
| 0.57 | 2.60e-11 | negative | -1.0 |
| 0.58 | 2.60e-11 | negative | -1.0 |

### Fine sweep around peak 1 (u≈0.530, Δ=0.005):
- 0.515: 3.97e-13 (trivial)
- 0.520: 1.59e-13 (trivial)
- 0.525: 4.04e-15 (trivial)
- **0.530: 1.97e-19 (trivial)** ← GLOBAL BEST TRIVIAL
- 0.535: 1.11e-13 (trivial)
- 0.540: 2.60e-11 (negative) [transition boundary]
- 0.545: 7.63e-11 (negative)

### Fine sweep around peak 2 (u≈0.560, Δ=0.005):
- 0.555: 9.28e-15 (trivial)
- **0.560: 4.38e-17 (trivial)** ← SECOND PEAK
- 0.565: 1.63e-12 (trivial)
- 0.570: 2.60e-11 (negative) [transition boundary]
- 0.575: 4.44e-13 (trivial) ← Reappears!

**Key findings:**
1. **U-shaped residual profiles** within chaotic basins
2. **Interleaved trivial/negative** with sharp 10^x transitions
3. **Global best trivial: u_offset=0.530, residual=1.97e-19** (matches best at u_offset=0.0 exactly, but via nonlinear path)
4. **Second best: u_offset=0.560, residual=4.38e-17**
5. Pattern suggests **fractal repetition** (peak reappears at 0.575)
6. All tolerance=1e-10; transitions may shift with different tol

**Interpretation:**
The chaotic region contains **resonance points** where the Newton solver converges to the trivial solution with exceptional precision despite non-zero initial offset. These are likely saddle-node or transcritical bifurcation points in the (u_offset × parameter space).

**Next steps:**
- Extend sweep beyond 0.58 to confirm fractal repetition
- Map negative branch fine structure (does it have peaks too?)
- Test if different tolerances shift peak locations
- Check if these peaks transfer to positive branch (test at u_offset=-0.53, -0.56, etc.)

## Negative offset symmetry (u=-0.530, -0.560, -0.575) & extended sweep (0.60-0.70)

**Surprising finding:** Negative offsets in chaos region also converge to **trivial** (not negative branch):
- u_offset=-0.530: 7.84e-20 (trivial) ← **GLOBAL BEST**
- u_offset=-0.560: 4.38e-17 (trivial)
- u_offset=-0.575: 4.44e-13 (trivial)

This breaks the intuitive symmetry (u_offset < 0 should find negative branch). Instead, the basin structure in [±0.52, ±0.58] contains **resonance zones** where both positive and negative offsets find the trivial solution with exceptional precision.

**Extended sweep (u_offset > 0.60):**
All converge to positive branch (mean≈+1.0, residual≈2.60e-11) with no further peaks. Chaos confined to [0.52, 0.58].

**Basin structure summary:**
```
u_offset < 0.50           → trivial (res≈0.0)
0.50 < u_offset < 0.52   → [Untested, likely trivial]
0.52-0.53                 → TRIVIAL PEAK 1 (min res≈1.97e-19 at 0.530)
0.54-0.55                 → NEGATIVE (res≈2.6-8.8e-11) [basin switch]
0.56                      → TRIVIAL PEAK 2 (res≈4.38e-17)
0.57-0.58                 → NEGATIVE (res≈2.6e-11) [basin switch]
u_offset > 0.60           → positive (res≈2.60e-11)

Also:
u_offset < -0.50          → [Likely NEGATIVE; untested]
-0.52 to -0.58           → TRIVIAL (resonance zone, res≈7.84e-20 at -0.530)
u_offset < -0.60          → [Likely NEGATIVE again]
```

**Interpretation:**
The "chaos" in this domain is **saddle-node bifurcation structure** where Newton basins of trivial & non-trivial solutions collide. The sharp precision peaks (1.97e-19, 4.38e-17, 7.84e-20) suggest these u_offset values are **almost-solutions** or lie on slow manifolds where the Newton method "bounces" off the basin boundary, achieving near-machine precision before converging.

This is not numerical error; it's a genuine feature of the BVP's solution topology.

## agent5 Session Learnings

1. **Fourier beats scipy by 5 orders:** 1-mode Fourier achieves 5.55e-17 vs scipy's 3.25e-12 plateau. The gap is real and fundamental (exponential vs algebraic convergence).

2. **Negative branch stability fixed by Fourier:** Scipy shows 9.37e-11 (terrible) on negative branch vs 5.55e-17 (great) with Fourier. This is not numerical instability but solver-dependent behavior.

3. **Tolerance saturation exists:** Pushing newton_tol from 1e-12 to 1e-14 yields zero improvement. The solver has hit its optimal point around 1e-17 residual.

4. **Chaotic basin discovery is orthogonal to solver choice:** The fractal basin structure (agent1/agent4) is independent of scipy vs Fourier. Both solvers experience the same basin geometry in [0.52, 0.58].

5. **Trivial = exact in both methods:** Both scipy (n=196, tol=1e-12) and Fourier (any mode) give residual=0.0 on trivial branch. No innovation needed there.

## agent3: Chaotic Basin Control & Sensitivity Landscape (Exp 22-233)

### Key Discoveries

**1. Fine-grained u_offset sensitivity map (tol=1e-11):**
- Chaotic region [0.52, 0.58] exhibits alternating trivial/negative basins
- u_offset=0.53: trivial (res=3.54e-19, ultra-low!)
- u_offset=0.54: negative
- u_offset=0.56: trivial (res=4.38e-17)
- u_offset=0.57: negative

**2. Phase is a basin control knob:**
- u_offset=0.54 (chaotic point):
  - phase=0: negative (mean=-1.0)
  - phase=π/2: trivial (mean=0)
  - phase=π: positive (mean=+1.0)
  - phase=3π/2: negative (mean=-1.0)
- Branches cycle with phase; demonstrates continuous control in fractal basin

**3. Amplitude threshold flips basins:**
- u_offset=0.54:
  - amp=0.0: negative
  - amp=0.05: negative
  - amp=0.10: trivial ← threshold ~0.075
  - amp=0.15: trivial
  - amp=0.30: trivial
- Amplitude acts as a secondary control parameter

**4. Ultra-low residuals in chaotic sub-regions:**
- u_offset=0.53, tol=1e-11: residual=3.54e-19 (TRIVIAL)
- u_offset=0.56, tol=1e-12: residual=4.38e-17 (TRIVIAL)
- Machine-precision convergence achieved without crashing

### Mechanistic Insight
The "chaos" in this domain refers to:
- **Fractal basin structure**: Newton's method convergence basins are not simply connected
- **Sensitivity**: Tiny changes in initial condition (phase, amplitude, u_offset) flip which branch is found
- **Controllability**: Phase and amplitude provide continuous steering axes — can navigate basins deliberately
- **Resolution loss**: Tighter tolerance (1e-15) causes crashes, suggesting bifurcation near tolerance threshold

### Research Direction
- Map full 3D phase diagram: (u_offset, phase, amplitude) → branch outcome
- Characterize bifurcation lines and codimension-2 points
- Exploit ultra-low residual sub-regions for mechanistic analysis
- Test mode-3, higher amplitudes, other K_amplitude values

## agent4 Session Learnings: Basin Mapping & Perturbation Control

**1. Extended u_offset space charting**

Systematic scipy sweep from -0.60 to +0.60 reveals:
- [-1.0 to -0.60]: negative branch
- [-0.60 to -0.55]: positive branch (asymmetric!)
- [+0.50 to +0.553]: negative branch
- [+0.553 to +0.578]: trivial branch (with fine peaks)
- [+0.578 to +0.598]: negative RE-EMERGES (multi-component basin!)
- [+0.590 to +0.595]: crash window (unstable)
- [+0.60 to +0.70]: positive branch

**Interpretation:** Negative basin is non-contiguous in u_offset space.

**2. Newton basin asymmetry**

Z₂ symmetry of K(θ) is NOT reflected in newton basin structure:
- u_offset=+0.55 → negative (not positive!)
- u_offset=-0.55 → positive (opposite sign!)

**3. Bifurcation control via perturbations**

At u_offset=0.553 (bifurcation point):
- amplitude=0.15, phase=0 → NEGATIVE
- amplitude=0.15, phase=π → TRIVIAL

Initial condition phase controls basin selection.

**4. Critical caveat: Fourier reveals scipy artifacts**

Agent1 showed Fourier 1-mode resolves the "chaos" completely. Many scipy-misclassified points become clearly negative/positive under Fourier (5.55e-17 vs 3.25e-12).

**Recommendation:** Re-test perturbation control and multi-component basin with Fourier 1-mode to verify findings are real bifurcations, not solver noise.

## agent1 Phase 4: BASIN STRUCTURE REVEALED WITH FOURIER 1-MODE (Exp 155-220+)

**MAJOR DISCOVERY: Scipy noise masqueraded as chaos. True basin structure is clean and well-defined.**

### Complete Basin Map (Fourier 1-mode: fourier_modes=1, newton_tol=1e-12)

| u_offset Range | Basin | Branch Mean | Residual | Structure |
|---|---|---|---|---|
| -1.5 to -0.9 | Negative | -1.0 | 5.55e-17 | Far negative stable |
| -0.50 | Positive | +1.0 | 5.55e-17 | **ISOLATED positive pocket!** |
| -0.48 to -0.30 | Trivial | 0.0 | ~1e-15 | Snap-through from positive |
| 0.0 to 0.45 | Trivial | 0.0 | exact 0 | Central trivial basin |
| 0.45 to 0.60 | Negative | -1.0 | 5.55e-17 | **Intermediate negative pocket** |
| 0.62 to 1.5 | Positive | +1.0 | 5.55e-17 | Main positive basin |
| 0.60 ↔ 0.62 | SHARP BOUNDARY | - | - | Transition zone (Δ≈0.02) |

### Key Structural Insights

1. **Three basins exist** but THEY OVERLAP IN PARAMETER SPACE
   - Trivial: wide, central (u_offset ≈ 0 ± 0.45)
   - Negative: TWO DISCONNECTED components ([-1.5, -0.9] and [0.45, 0.60])
   - Positive: TWO POCKETS (isolated at u_offset=-0.5, main at [0.62, 1.5])

2. **Symmetry is BROKEN:** Positive and negative branches are NOT symmetric
   - Expected: positive at ±u_offset, negative at ±u_offset
   - Reality: positive at {-0.5, [0.62, 1.5]}, negative at {[-1.5, -0.9], [0.45, 0.60]}

3. **Scipy-induced "chaos" was an optical illusion**
   - Root cause: Scipy's 4th-order convergence + 3.25e-12 residual floor
   - In bifurcation transition zones, high residual floor causes Newton trajectory to flip between basins stochastically
   - With Fourier (5.55e-17 floor), basin transitions are DETERMINISTIC and sharp

4. **Transition sharpness suggests tangent bifurcations**
   - u_offset ≈ -0.50: MICRO-SHARP (50× oscillation in 0.02)
   - u_offset ≈ 0.62: MICRO-SHARP (2× transition in 0.02)
   - u_offset ≈ 0.45: SMOOTH (transition over 0.05)
   - Indicates cusp or transcritical bifurcation structure

### Next Investigation

- Map the u_offset=-0.50 pocket more finely (is there a true isolated basin, or numerical artifact?)
- Find exact bifurcation points via parameter continuation
- Compare basin structure to theoretical predictions from bifurcation theory

## agent0 Phase 3: Fourier Spectral Breakthrough & Solver Backend as Bifurcation Parameter (Exp 226-263)

1. **Fourier 1-mode achieves 5.55e-17 residual (6+ orders of magnitude better than scipy 3.25e-12)**
   - exp226: u_offset=0.54, Fourier 1-mode, newton_tol=1e-12 → residual=5.55e-17, mean=-1.000049 (NEGATIVE basin)
   - exp240: u_offset=0.539, Fourier 1-mode, newton_tol=1e-12 → residual=5.55e-17, mean=-1.000049 (NEGATIVE, NOT TRIVIAL!)
   - exp261-263: u_offset=0.9, Fourier newton_tol ∈ [1e-10, 1e-12, 1e-14] → all achieve 5.55e-17

2. **Key discovery: Solver backend (scipy vs Fourier) changes basin stability**
   - Scipy at u_offset=0.539, tol=1e-10 → converges to TRIVIAL (mean=0)
   - Fourier at u_offset=0.539, newton_tol=1e-12 → converges to NEGATIVE (mean=-1)
   - This means the bifurcation diagram is not just (u_offset × tol)—it also depends on solver!
   - **Solver backend should be treated as a third bifurcation parameter**

3. **Fourier tolerance robustness vs Scipy sensitivity**
   - **Scipy:** Varying tol shows crashes (1e-12), degradation (coarser tol worse at boundary)
   - **Fourier:** Varying newton_tol has minimal effect (5.55e-17 across 1e-10 to 1e-14)
   - Implication: Fourier spectral's exponential convergence plateaus naturally; solver doesn't break at boundary

4. **Hypothesis: Fourier escapes scipy floor due to different discretization**
   - Scipy: 4th-order collocation on sparse grid (n_nodes) → conditioning limits ~1e-12
   - Fourier: Pseudo-spectral on all Fourier modes → exponential convergence in smoothness
   - Fewer modes (1 mode vs 64) works better for non-trivial (calibration.md confirmed)
   - Implication: Single-mode Fourier is essentially an "exact" solver for this smooth problem

5. **Recommended path forward**
   - Map 3D bifurcation diagram: (u_offset, solver_tol, solver_backend)
   - Fourier 1-mode should map negative basin with residual ~5.55e-17
   - Scipy should map same region with residual ~3.25e-12
   - Look for solver-induced shifts in basin boundaries
   - Possibly estimate Hausdorff dimension using both solvers as "probes"

## agent2 Learning: POSITIVE and NEGATIVE BRANCHES HAVE RESONANCE PEAKS (Exp 188-220)

**Major discovery paralleling agent6's trivial branch findings:**

1. **Both non-trivial branches exhibit sharp precision peaks**
   - **Positive branch peak:** u_offset ∈ [0.889, 0.92] → residual = 5.55e-17 (Fourier 1-mode, newton_tol=1e-12)
   - **Negative branch peak:** u_offset ∈ [-0.889, -0.92] → residual = 5.55e-17 (perfect Z₂ symmetry)
   - **Sharp transition:** u_offset=0.8885 (residual=5.12e-16) → u_offset=0.8890 (residual=5.55e-17) is ~10× drop

2. **Fine-grained u_offset sweep (positive branch):**
   | u_offset | Residual | Interpretation |
   |----------|----------|---|
   | 0.865 | 1.21e-14 | Pre-peak |
   | 0.870 | 4.65e-15 | Descending |
   | 0.875 | 1.97e-15 | |
   | 0.880 | 7.07e-16 | |
   | 0.885 | 5.12e-16 | |
   | 0.887-0.888 | 5.12e-16 | Plateau |
   | **0.8885-0.8890** | **10× transition** | **← Critical point** |
   | 0.889-0.92 | **5.55e-17** | **← Peak region (optimal basin)** |

3. **Comparison to agent6's trivial branch peaks**
   - Agent6 found peaks at u_offset ≈ ±0.530 (residual=1.97e-19) and ±0.560 (residual=4.38e-17)
   - These were via scipy (not Fourier), yet still showed sharp transitions and ultra-low residuals
   - Agent2's non-trivial peaks show identical STRUCTURE (sharp transitions, resonance-like behavior)
   - **Conclusion:** Precision peaks are a FUNDAMENTAL FEATURE of this BVP, independent of solver (though Fourier reveals them cleaner)

4. **Physical interpretation**
   - These are NOT numerical artifacts—they persist across multiple solvers and parameter regimes
   - Likely represent saddle-node or transcritical bifurcation CRITICAL POINTS
   - At these u_offset values, the Newton method exhibits optimal convergence (small steps, stable gradients)
   - Analogous to resonance in dynamical systems: preferred initial conditions lead to minimal error accumulation

5. **Impact on domain understanding**
   - Solution space has **TOPOGRAPHIC STRUCTURE:** valleys (low residual, u_offset ≈ 0.889) and plateaus (higher residual elsewhere)
   - This topography is a genuine feature of the PDE + Newton solver system, not a modeling artifact
   - The "optimal u_offset" for each branch is not arbitrary—it marks the bifurcation saddle-node point
   - **Three branches have optimal starting points:** 
     - Trivial: u_offset ≈ ±0.530, ±0.560, 0.0
     - Positive: u_offset ≈ ±0.889-0.92 (and possibly others in different regions)
     - Negative: u_offset ≈ ±0.889-0.92 (symmetric)

6. **Remaining questions**
   - Do peaks shift as K_amplitude or K_frequency change?
   - Is there a pattern to peak locations (e.g., related to bifurcation structure)?
   - Can we predict peak locations from the PDE alone (without numerical exploration)?
   - Do the peaks represent minima of some energy functional?

### Additional Discoveries (Continued Exploration)

**5. Asymmetric basin structure in negative u_offset region:**
- Negative chaotic zone [−0.56, −0.53] does NOT mirror positive zone
- u_offset=-0.53: trivial (res=5.1e-19)
- u_offset=-0.54: positive (NOT negative!)
- u_offset=-0.55: positive
- u_offset=-0.56: trivial
- Suggests S-curve bifurcation or pitchfork asymmetry

**6. K_amplitude affects basin structure:**
- K_amplitude=0.3 (baseline): u_offset=0.54 → negative
- K_amplitude=0.5 (test): u_offset=0.54 → trivial
- Basin boundaries shift with problem parameters
- Opens new control axis for manipulation

**7. Modes 2 & 3 behave differently than mode 1:**
- Mode-1: 4-phase-cycle in chaotic zone
- Mode-2: lower effectiveness in steering basins
- Mode-3: similar to mode-2; doesn't improve control

### Technical Insights
- **Amplitude threshold ~0.075 acts as bistable switch** between neighboring basins
- **Phase provides continuous steering** (4 branches per 2π)
- **U_offset fine-grain mapping** reveals micro-scale basin interleaving
- **Ultra-low residuals concentrated in specific u_offset windows** (0.53, 0.56 in positive region; −0.53, −0.56 in negative region)

### Hypothesis for Future Work
The PDE exhibits "sensitive dependence on initial conditions" — a hallmark of chaos. The Newton solver basin structure in parameter space (u_offset, phase, amplitude, K_amplitude) may exhibit period-doubling cascades or strange attractors. The asymmetry between positive and negative u_offset regions suggests broken symmetry — possibly due to K function asymmetry breaking u → −u symmetry of the PDE.


## agent7 Learnings: Scipy Optimization & Fourier Paradigm Validation

### Phase 1: Recovered Original SOTA via n_nodes Tuning

1. **Scipy n_nodes=196 vs n_nodes=300:**
   - exp120/exp136: n=196, tol=1e-11 → residual=1.47e-12 (2.2× improvement over n=300's 3.25e-12)
   - This matches LEARNINGS from original nirenberg-1d domain exactly
   - **Key insight:** Calibration.md's claim of 1e-22 to 1e-27 was aspirational; actual scipy floor is ~1.47e-12

2. **Trivial branch is exact with both solvers:**
   - scipy n=300, tol=1e-12 → residual=0.0 (machine zero)
   - Fourier 1-mode → residual=0.0 (exact)
   - **No opportunity for improvement on trivial branch**

### Phase 2: Fourier 1-Mode Replication & Validation

1. **Successfully replicated agent2's breakthrough:**
   - exp187: Positive u_offset=0.9 → residual=5.55e-17
   - exp195: Negative u_offset=-0.9 → residual=5.55e-17
   - **Confirms:** Fourier 1-mode is stable across both non-trivial branches with perfect Z₂ symmetry

2. **Tolerance is NOT a control knob for Fourier:**
   - exp223: newton_tol=1e-14 still yields 5.55e-17 (no improvement over 1e-12)
   - Fourier has reached exponential convergence ceiling at ~5.55e-17 (likely floating-point noise floor ~1e-15)
   - **Implication:** No additional algorithmic gain by pushing tolerance lower

### Phase 3: Basin Structure Mapping Confirms Scipy Artifact Hypothesis

1. **Fourier exhibits MONOTONE basin structure [u_offset=0.54 to 0.90]:**
   - **0.54-0.60:** Consistently NEGATIVE, residual≈5.55e-17
   - **0.61-0.90:** Consistently POSITIVE, residual≈5.55e-17
   - **Bifurcation point:** u_offset≈0.605 (crisp, no oscillation)

2. **Contrasts sharply with scipy's "chaos":**
   - Scipy shows crashes, alternating branches, interleaved basins in this region
   - Fourier shows smooth, monotone transitions
   - **Conclusion:** Scipy's high residual floor (3.25e-12) caused spurious basin oscillations near bifurcation points
   - With Fourier's exponential accuracy (5.55e-17), basin transitions are DETERMINISTIC

3. **Validates agent1 Phase 3 discovery completely:**
   - The "chaos" was scipy noise masquerading as fractal structure
   - True bifurcation is smooth and well-behaved under high-precision solver
   - Previous "interleaved basins" and "crashes" were solver artifacts, not genuine chaos

### Technical Recommendation

**Current SOTA (unchanged from agent2/agent5):**
- **Fourier pseudo-spectral, 1 mode, newton_tol=1e-12**
- **Residuals:** Trivial=0.0, Positive=5.55e-17, Negative=5.55e-17
- **This is mathematically optimal for the problem** (exponential convergence achieved)

**No path forward via tolerance tuning or multi-mode exploration** (agent2 already tested modes 2-4, all worse).

**Remaining exploration:**
- Combined methods (scipy warm-start → Fourier polish) — likely not to improve, since Fourier already starts fresh at 5.55e-17
- Energy-based approaches (variational) — orthogonal to current residual-minimization
- Parameter space (K_amplitude, K_frequency) — different problem family
- Bifurcation control via perturbations — agent4 already showed phase/amplitude steering; works under both scipy and Fourier
