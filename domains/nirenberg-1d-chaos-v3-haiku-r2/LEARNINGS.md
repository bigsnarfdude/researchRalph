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
