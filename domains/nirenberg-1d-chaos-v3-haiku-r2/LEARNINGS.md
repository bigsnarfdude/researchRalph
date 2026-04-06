## Learning Log — agent2

### Phase 1: Branch Discovery & Robustness (Exps 003-006, 080-091)
- **Finding**: All three branches (trivial, positive, negative) easily accessible via u_offset parameter
- **Robustness**: Stable across amplitude (0.0-0.5), modes (1-3), phase shifts (0-2π)
- **Phase stability**: Phase shifts do not flip branches when u_offset is properly set
- **Convergence**: Both branches converge monotonically with tighter tolerance or finer mesh

### Phase 2: Positive vs Negative Convergence (Exps 80-91, 92-101)

**Positive branch (u_offset=0.9):**
- n_nodes sweep: monotonic improvement (5.9e-9 → 2.1e-10)
- Tolerance sweep: clean linear convergence in log scale

**Negative branch (u_offset=-0.9):**
- Initial n_nodes sweep appeared oscillatory (non-monotonic)
- Root cause: mixed solver parameters (fixed tol=1e-8 with varying n_nodes)
- True behavior: clean convergence with appropriate tolerance
- **Conclusion**: Both branches are mathematically equivalent; apparent asymmetry was an artifact

### Phase 3: Branch Boundaries (Exps 102-150)
**Precise u_offset mapping:**
- Negative branch: u_offset ≤ -0.70 (sharp boundary)
- Trivial branch: -0.70 < u_offset < 0.54 AND 0.56 < u_offset < 0.58 (two islands!)
- **Hidden window**: u_offset = 0.55 → negative branch (isolated, ~0.01 width)
- Positive branch: u_offset ≥ 0.60 (sharp boundary at ~0.58)

### Phase 4: The u_offset=0.55 Bistable Window (Exps 151-270)

**Discovery**: Found isolated region where solver preferentially finds negative solution instead of trivial/positive.

**Characterization:**
- Width: approximately ±0.01 around u_offset=0.55
- Mean: -1.000218, residual: 5.67e-9 (consistent with negative branch)
- Robust across tolerances (1e-6 to 1e-10) — NOT solver artifact
- Robust across mesh sizes — INTRINSIC to problem

**Amplitude sensitivity:**
- amplitude ≤ 0.1: stays in negative basin
- amplitude ≥ 0.2: escapes to trivial branch
- **Mechanism**: Low-amplitude perturbations preferentially find negative solution

**Mode sensitivity:**
- n_mode=1: sustains negative basin (amplitude ≤ 0.1)
- n_mode ≥ 2: escapes to trivial branch (even with amplitude=0.1)
- **Mechanism**: Mode-1 resonance with K function at this u_offset

**Symmetry test:**
- No mirror window at u_offset=-0.55 (tested amplitudes 0-0.2, modes 1-2)
- Bifurcation structure appears **asymmetric** in parameter space

### Key Insights

1. **Solution space is richer than initially described**: Three "canonical" branches exist, but there are also isolated bistable windows that can be accessed with specific (u_offset, amplitude, mode) combinations.

2. **Basin of attraction is parameter-dependent**: Initial condition properties (amplitude, mode) determine which branch the solver finds. This is classic bifurcation behavior.

3. **Resonance phenomenon**: The u_offset=0.55 window appears to be a resonance between the u_offset parameter and the K(θ) function structure.

4. **No hidden higher branches found**: Tested modes 1-6; no evidence of additional solution branches beyond the known three (and the 0.55 window).

5. **Mathematical asymmetry**: Positive and negative branches behave identically under solver refinement, but the bifurcation structure asymmetrically favors finding the negative solution at a specific u_offset.

### Questions for Future Work
- Are there similar windows elsewhere in (u_offset, amplitude, mode) space? (Partial scan suggests isolated)
- What is the mathematical origin of the u_offset=0.55 resonance? (Likely related to K_amplitude=0.3 and K_frequency=1)
- Can we control which branch is found via phase shifts? (Preliminary tests suggest phase has minimal effect on branches, unlike amplitude/mode)
- Do windows exist when K_amplitude or K_frequency are varied?

## Learning Log — agent3

### Optimization Breakthrough: Amplitude + Solver Interaction (Exps 008-011, 016-022, 049-077)

**Discovery**: Trivial branch residual can be dramatically improved via amplitude tuning + tight solver tolerance.

**Amplitude sweep on trivial branch (u_offset=0.0, n_nodes=100, tol=1e-8):**
- amplitude=0.0: residual=5.6e-11
- amplitude=0.1: residual=2.3e-12 ← first big jump!
- amplitude=0.2: residual=1.5e-11 (regression)
- amplitude=0.3: residual=1.2e-13 ← **OPTIMAL** (42x better than baseline!)
- amplitude=0.4: residual=1.9e-10 (steep regression)

**Ultra-fine settings on trivial (amplitude=0.3, n_nodes=200, tol=1e-12):**
- Residual = 6.55e-18 (well below baseline 5.6e-11)
- Near machine precision limit for residuals of O(1)
- **Magnitude improvement**: 10,000x vs agent1's best (2.98e-13)

**Positive/negative branch optimization (n_nodes=200, tol=1e-11):**
- Both branches achieve residual ≈ 1.0e-11 (vs baseline 2.4-5.7e-9)
- Stability limit: cannot go below tol=1e-11 without crashes
- Non-trivial branches are numerically "stiffer" than trivial

### Key Discoveries

1. **Branch-specific convergence**: Trivial branch is numerically well-conditioned (tol down to 1e-12), while ±1 branches plateau at 1e-11. Suggests different mathematical structure (trivial is simpler/degenerate).

2. **Amplitude creates optimal "roughness"**: Amplitude=0.3 is a sweet spot—too smooth (amplitude→0) loses convergence, too rough (amplitude>0.4) diverges. Suggests Fourier content in initial guess helps nonlinear solver navigation.

3. **Phase variations are branch-specific**: 
   - Trivial: phase shifts change residuals (up to 1.1e-22 reported by agent2)
   - Non-trivial (±1): phase shifts have NO effect (noise at ±1% level)
   - **Hypothesis**: Phase matters only when solution is degenerate/weakly nonlinear; breaks down for stable fixed points.

4. **Bifurcation structure is complex and not well-predicted by initial-condition parameters alone**. Agent0's fine-grained mapping revealed three sharp transitions, not smooth curves.

## Learning Log — agent0

### Phase 1: Baseline Branch Confirmation & Amplitude Mapping (Exps 001-062)

**Baseline discovery (amplitude=0, phase=0):**
- Trivial (u_offset=0.0): residual=5.64e-11, mean=0.0
- Positive (u_offset=0.9): residual=5.73e-9, mean=+1.0
- Negative (u_offset=-0.9): residual=2.42e-9, mean=-1.0

**Amplitude sweep on positive branch (u_offset=0.9):**
- amplitude=0.1: residual=5.73e-9 (baseline)
- amplitude=0.3: residual=2.42e-9 ← **OPTIMAL** (2.4x better!)
- amplitude=0.45: residual=5.64e-9 (regression)

**Fourier mode testing on positive (u_offset=0.9, amplitude=0.3):**
- mode=1: residual=2.42e-9
- mode=2: residual=2.42e-9 (identical!)
- mode=3: residual=2.42e-9 (identical!)
- **Finding**: Mode is irrelevant; amplitude is the control parameter.

### Phase 2: U-Offset Bifurcation Boundary Mapping (Exps 067-117)

**Sharp bifurcation at u_offset ≈ 0.495-0.50:**
- u_offset=0.0-0.49: all trivial (residuals 5e-11 to 5e-20, excellent!)
- u_offset=0.50: NEGATIVE (residual=2.42e-9, mean=-1.0) ← sudden jump
- u_offset=0.60: positive (residual=5.73e-9, mean=+1.0)

**Negative side asymmetry:**
- u_offset=-0.3: trivial
- u_offset=-0.5: **POSITIVE** (mean=+1.0, residual=2.42e-9) ← opposite of +0.5!
- u_offset=-0.9: negative

**Interpretation**: Bifurcation structure is asymmetric. The "mirror" of u_offset=0.50→negative is u_offset=-0.50→positive, NOT u_offset=-0.50→negative.

### Phase 3: CRITICAL DISCOVERY — Bifurcation Manifold Structure (Exps 188-257)

**The primary bifurcation at u_offset=0.50 is MULTI-DIMENSIONAL!**

**Direct bifurcation control via amplitude at u_offset=0.50:**
- amplitude=0.0, phase=0: NEGATIVE (exp070, residual=2.42e-9, mean=-1.0)
- amplitude=0.1, phase=0: TRIVIAL (exp224, residual=5.17e-11, mean=0.0) ← **MANIFEST SHIFT!**
- amplitude=0.1, phase=π: TRIVIAL (exp217, residual=1.31e-11, mean=0.0)
- amplitude=-0.1, phase=0: TRIVIAL (exp230, residual=1.31e-11, mean=0.0)

**Control away from bifurcation:**
- u_offset=0.60 + amplitude=0.1: POSITIVE (exp238, residual=5.73e-9, mean=+1.0) ← amplitude inert
- u_offset=0.4975 + amplitude=0.0: TRIVIAL (exp257, residual=5.55e-11, mean=0.0) ← before transition

**Key implications:**
1. **Phase is ORTHOGONAL to branch selection**: phase=0 and phase=π give identical outcomes when amplitude is present
2. **Amplitude acts as bifurcation control parameter**: ∃ critical amplitude≈ε (small) that shifts u_offset=0.50 from negative to trivial
3. **Effect is localized to bifurcation**: Away from critical u_offset, amplitude has no effect on branch selection
4. **This CONTRADICTS earlier claim** that "amplitude is irrelevant"—it's irrelevant AWAY from bifurcation but CRITICAL AT boundaries

### Phase 4: Bifurcation Manifold Characterization

**Conjecture**: The bifurcation structure is (at least) 2-dimensional: (u_offset, amplitude).

- **Codimension-2 bifurcation structure**: u_offset controls primary branch, amplitude modulates selection at critical u_offset
- **Phase is orthogonal degree of freedom**: does not affect which attractor is selected
- **Critical region**: u_offset ≈ 0.495-0.50 with amplitude ≈ 0-0.1 separates negative from trivial basins

**Comparison with agent2/agent3 findings:**
- Agent2 found u_offset=0.55 window with amplitude sensitivity (amplitude ≤ 0.1 stays negative, ≥ 0.2 escapes)
- Agent3 found amplitude=0.3 optimizes trivial residuals (42x improvement)
- **Agent0 finding**: Amplitude controls which BRANCH is accessed at bifurcation (not just residual magnitude)

**New view of solution space**: Not "three isolated branches" but a **manifold of solutions parameterized by (u_offset, amplitude, mode, phase)** with complex basin structure. The three "canonical" branches are stable fixed points, but approach to bifurcation boundaries reveals rich structure controlled by perturbation parameters.

### Key Questions

1. What is the precise amplitude threshold at u_offset=0.50 that separates negative/trivial basins?
2. Do similar manifold structures exist at other bifurcation boundaries (±0.55, ±0.6)?
3. Can phase shifts be used to navigate bifurcation manifold in regions where amplitude doesn't work?
4. What is the mathematical mechanism? (Likely: K function resonance + nonlinear PDE dynamics)

## Phase 5: Amplitude-Dependent Bifurcation Cascades (Exps 288-298)

**New Discovery**: At u_offset=0.55, the basin structure is amplitude-dependent in an **oscillatory** pattern:

- amplitude ∈ [0.00]: Negative basin
- amplitude ∈ [0.02-0.08]: Trivial basin
- amplitude ∈ [0.10-0.12]: Negative basin (returns!)
- amplitude ∈ [0.14+]: Trivial basin

**Mechanism**: This is NOT a simple threshold. Multiple stable basins coexist, and their relative attractiveness oscillates with amplitude. Characteristic of nonlinear bifurcations.

**Implication**: The solution landscape is **far richer** than three branches. There's a bifurcation manifold in (u_offset, amplitude) space with complex basin structure.

## Summary Status: INCOMPLETE EXPLORATION

**What we know:**
- Three canonical branches exist and are robust
- Bifurcation manifold exists around u_offset ≈ 0.55
- Basin structure is amplitude-dependent (nonlinear)
- Structure may be period-doubling or chaotic (oscillatory behavior)
- No other isolated windows found in coarse scans

**What remains uncertain:**
- Full extent of bifurcation manifold (other u_offset values?)
- Nature of bifurcation (period-doubling, cusp, chaos?)
- Role of phase and higher modes at bifurcation
- Mathematical origin (resonance with K function)

**Recommendation**: 
1. Fine-map amplitude dependence across u_offset = 0.50-0.60 range
2. Test phase effects at bifurcation points
3. Investigate mode 2, 3 behavior at bifurcation
4. (Future) Perturb K_amplitude or K_frequency to validate resonance hypothesis

**Confidence level**: High for discovered phenomena (reproducible, robust), Medium for interpretation (need deeper bifurcation theory analysis).

