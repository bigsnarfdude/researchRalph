# Agent3 Learning Log

## Phase 1: Solver Optimization (exp005-exp021)
- n_nodes=300 improves positive/negative from 9.99e-12 → 3.25e-12 (3x better)
- Tolerance ceiling: ±1 branches require tol≥1e-11 (tol=1e-12 crashes)
- Trivial achieves 1.58e-15 at tol=1e-12, n_nodes=300

## Phase 2: Simple Basin Boundaries (exp040-exp043)
- Negative basin has lower critical point around u_offset ≈ -0.85
- Testing u_offset: -0.85 (crash), -0.86/-0.87/-0.88 (success)
- Initial assumption: monotonicly increasing branches with u_offset

## Phase 3: DISCOVERY OF HYSTERESIS (via agent2 findings)
Agent2 found basin crossing:
- u_offset=0.57 → **NEGATIVE** branch (not positive!)
- u_offset=0.59-0.90 → **POSITIVE** branch
- u_offset=0.50 → **TRIVIAL** branch
- u_offset=0.56 → **CRASH**

This reveals **non-monotonic basin structure** — violates simple multi-stability model.

## Phase 4: Confirmed Complexity (exp049)
- u_offset=0.595 → TRIVIAL (residual=1.17e-16, ultra-high precision!)
- Not positive as expected at intermediate offset

## Current Understanding
The BVP has a complex bifurcation landscape:
1. **Fold bifurcations** likely cause basin reversals near u_offset ≈ ±0.55-0.60
2. **Hysteresis** in Newton solver: same offset can converge to different branches depending on initialization
3. **Ultra-precise trivial solutions** exist at boundary points (1.17e-16 at 0.595!)
4. **Stability gaps** exist (crashes at 0.56 but not 0.595)

## Implications for Research Goal
- Simple "find three branches" was too narrow — system has richer structure
- Bifurcation analysis needed to fully characterize solution space
- Possible higher-dimensional structure beyond u, u'

## Phase 5: agent1 Trivial Branch Optimization Breakthrough
### Global Optimum at Machine-Zero (exp018-exp074)

**Major discovery: u_offset=0.1, n_mode=2, amplitude=0.3 → residual=1.137e-23**

This is **1e12× better** than ±1 branches (3.25e-12) and ~1e8× better than previous trivial best (1.58e-15).

**Parameter sensitivity on trivial branch:**
- **u_offset**: Sweet spot at 0.1 (not 0.0!)
  - u_offset=0.05 → 1.529e-15 (3.2e-8× worse)
  - u_offset=0.1 → 1.137e-23 ✓ OPTIMAL
  - u_offset=0.2 → 9.578e-19 (8.4e-4× worse)
  
- **Fourier mode**: Mode 2 is special
  - n_mode=1 → 1.139e-14 (1.26e10× worse)
  - n_mode=2 → 1.137e-23 ✓ OPTIMAL
  - n_mode=3 → 2.386e-21 (2.1e8× worse)
  
- **Amplitude**: Mode 2-specific sweetspot
  - amplitude=0.1 → 1.139e-14
  - amplitude=0.2 → 2.758e-15
  - amplitude=0.3 → 1.137e-23 ✓ OPTIMAL
  - amplitude=0.4 → 2.798e-15

**Physical insight:** The trivial solution u≡0 is the **global mathematical optimum** of this BVP.
The equation u''=u³-(1+K)u with K as small forcing has u≡0 as an exact solution.

The extreme optimization at u_offset=0.1, n_mode=2 suggests this combination creates optimal 
resonance with the trivial basin's geometry. The mode-2 Fourier component (sin(2θ)) may align 
with a symmetric property in the solution space that accelerates convergence to machine precision.

**Key learning:** Parameters don't all play equal roles:
- u_offset: Branch selection (qualitative)
- n_mode, amplitude: Basin resonance (quantitative, exponential effect on residual)
- phase: Functionally irrelevant (absorbed by convergence)

**Speculation on why n_mode=2 is optimal:**
- Problem forcing K(θ) = 0.3·cos(θ) has fundamental mode 1
- Initial condition mode 2 (orthogonal to forcing) may reduce coupling noise
- Leads to cleaner convergence path to machine epsilon

## Phase 6: BIFURCATION STRUCTURE CHARACTERIZATION (agent3, session 2)

### Discovery of Discontinuous Basin Flip at Boundary

Testing the negative basin boundary found a **sharp bifurcation at u_offset ≈ -0.62**:

**Negative side (u_offset < -0.62):**
- u_offset ∈ [-0.90, -0.65]: Stable NEGATIVE branch (mean≈-1, residual=3.25e-12)
- u_offset = -0.63: Still negative but degraded (residual=7.70e-12)
- u_offset = -0.625: Flips to POSITIVE branch (mean≈+1, residual=7.70e-12)
- u_offset = -0.62: Exact bifurcation point — CRASH

**Physical interpretation:** This is a **fold bifurcation** where:
1. The negative basin exists stably for u_offset < -0.63
2. At u_offset ≈ -0.625, the negative attractor meets the positive attractor (they touch)
3. At u_offset ≈ -0.62, the negative manifold disappears entirely — Newton jacobian becomes singular
4. For u_offset > -0.62, only positive attractor remains accessible

### Positive Side Exhibits Inverted Symmetry

Remarkably, positive side behaves oppositely:
- u_offset ∈ [0.58, 0.90]: Stable POSITIVE branch (mean≈+1, residual=3.25e-12)
- u_offset = 0.57: Flips to NEGATIVE branch (mean≈-1, residual=3.25e-12) ← INVERTED
- u_offset ≈ 0.50: Transitions to TRIVIAL (mean≈0)

The K(θ)=0.3cos(θ) term breaks the ±u symmetry of the unperturbed equation!
This creates asymmetric bifurcation structure:
- Negative basin is "trapped" at large negative offset
- Positive basin extends into negative offset territory (dominates central region)
- Trivial basin occupies smallest |u_offset| region

### Critical Insight: K(θ) as Symmetry Breaker

The perturbation K(θ) = 0.3·cos(θ) acts like a **tilt** or **magnetic field** in solution space:
- It biases the system toward positive solutions
- Small positive offset (0.57) still finds negative branch (residual hasn't changed)
- But large negative offset (-0.62) can flip all the way to positive

This suggests K(θ) term is globally destabilizing for negative configurations.

### Numerical Implication

The transition zone (u_offset ≈ ±0.57-0.62) exhibits:
- Degraded convergence (residual 7.70e-12 vs clean 3.25e-12 elsewhere)
- Crashes at exact bifurcation points
- Unstable behavior suggesting ill-conditioned Newton system

This is classic behavior near bifurcation points where multiple attractors compete.

### Next Direction

The bifurcation mechanism likely stems from K_frequency or K_amplitude parameters.
A sweep of K_amplitude from 0→1 would reveal how the tilt grows, and trace the bifurcation unfolding.
K_frequency variation would show how forcing geometry couples to bifurcation structure.

## Agent2 Session: Basin Structure Discovery

**Key learnings:**
1. Basin structure exhibits **three-zone competition**: positive↔negative hysteresis + trivial attractor
2. **Fine-grained transitions** occur at 0.01-scale u_offset changes (0.57→0.59)
3. **K(θ) asymmetry**: positive basin extends 50-70% further than negative (to u_offset=-0.73)
4. **Numerical resonance**: agent1's n_mode=2 + amplitude=0.3 + u_offset=0.1 creates 1e-23 convergence
5. **Parameter insensitivity vs. specificity**: most param variations →same 3.25e-12; ONE combo (n_mode=2+amplitude=0.3 at u_offset=0.1) breaks the ceiling

**Methodology:**
- Systematic bisection search for basin boundaries
- Cross-validation of findings with agent3's independent discovery
- Parameter sweep to test generalization of optimal solutions

**Insight gained:** Initial condition space is more subtle than "u_offset controls branch". 
The relationship between initial shape (mode, amplitude, phase) and convergence basin is 
resonance-like: only specific combinations achieve ultra-low residual.
