# Meta-Blackboard — nirenberg-1d-chaos-haiku-nigel-h2

## Current best
**Score: 0.0** (trivial branch, exp001)  
**Config:** `u_offset=0.0, amplitude=0.0, n_mode=1, phase=0.0`  
*Confidence: high. This is a perfect solution (u ≡ 0) and is reproducible.*

However: **all non-trivial solutions plateau at ~2.67e-13 (positive/negative branches, exp002-061).** This is within floating-point noise (1e-13 ~ machine epsilon for float64). The trivial solution is not meaningfully "better" — it's exact by mathematical design.

---

## What works (ranked by reproducibility)
1. **Trivial branch (u_offset=0.0)** — score 0.0, reproducible. But mathematically degenerate (u=0 is always a solution).
2. **Positive branch (u_offset≈+0.9)** — score ~2.67e-13, reproducible across 20+ trials. Real solution with mean ≈ +1.
3. **Negative branch (u_offset≈-0.9)** — score ~2.67e-13, reproducible across 20+ trials. Real solution with mean ≈ -1.
4. **Fractal basin boundaries** (exp001-035) — agents discovered that solution branches interleave chaotically across u_offset space (0.47→pos, 0.48→pos, 0.49→neg, ...). Interesting dynamically, but doesn't improve score.

---

## Dead ends (all plateau at 1e-13)
**Solver precision sweeps** (exp058-061): newton_tol ∈ {1e-8, 1e-10, 1e-12, 1e-14} — no lift. Same residual.

**Phase shifts** (exp053-057): phase ∈ {0, π/4, π/2, π, 1.5π, 2π} — no improvement.

**Amplitude variations** (exp041-045, 032): amp ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3} — plateau.

**Fourier mode sweeps** (exp015-016, 040, 047-051): modes ∈ {1, 2, 3, 32, 64} — plateau. (128, 256 modes crashed.)

**Agent0 design** (47 experiments, 0 keeps): exhausted parameter space, hit 0-keep wall.

**Agent1 design** (14 experiments, 0 keeps): same outcome.

---

## Patterns noticed
- **Numerical plateau at 1e-13:** 55 consecutive experiments (exp006-061) within 4× of each other. This is the floating-point precision floor for this solver, not a genuine search space.
- **Redundancy explosion:** After exp005, agents cycled through solver tweaks (tol, modes, phase, amp) that are mathematically equivalent given the plateau. No new degrees of freedom explored after basin boundaries.
- **Fractal discovery not exploited:** Agents found chaotic basin interleaving (exp001-035) but then switched to solver tuning instead of investigating *why* branches interleave or *whether* this chaotic structure encodes other solutions.
- **Both agent designs at 0-keep:** Suggests the scaffold (program.md) has exhausted its value, not that agents are incapable.
- **5 crashes** (exp011, 014, 017-018, 026): Fourier modes=128/256 likely triggered numerical instability or memory issues.

---

## Blind spots
- **Are positive/negative branches "solutions" or numerical artifacts?** All three branches converge to residual ≈ e-13. Is this a fundamental symmetry or measurement error?
- **Chaotic basin structure unexplored:** The fractal interleaving is a dynamical signature. What happens if you *exploit* chaos rather than *suppress* it?
- **No alternative metrics:** All 61 experiments optimize residual. No attempts at Sobolev norm, stability measures, or branch distance (how different are +1 and -1 branches really?).
- **No problem reformulation:** Agents treat Nirenberg as a fixed inverse problem. Never questioned whether residual is the right objective for "solving chaos."
- **Singular/bifurcation analysis:** The solution branches exist; no one asked *why* (bifurcation diagrams, continuation methods, or K parameter sweeps).

---

## Stepping stones
- **Fractal basin boundaries (exp001-035):** Agents discovered non-monotonic branch selection. This is surprising and suggests chaotic basin structure. Not directly improving score, but could lead to new problem formulations (e.g., "find all basins," "characterize chaos").
- **Mode-2 and mode-3 perturbations** (exp039): Slightly different residuals (3.25e-13, 3.92e-13) compared to baseline. Negligible gain, but suggests modal structure matters *slightly*. Worth one follow-up if problem is reformulated.

---

## Surprises
- **Expected:** Positive and negative branches would be "better" (lower residual) than trivial.  
  **Actual:** All three branches achieve indistinguishable residuals (within 4×10⁻¹³).  
  **Why:** Floating-point precision floor. Solver convergence at machine epsilon, not algorithmic limit.

- **Expected:** Tighter Newton tolerance or higher Fourier modes would improve score.  
  **Actual:** No improvement (exp058-061, 015-016, 047-051).  
  **Why:** Already at numerical precision limit. Solver can't improve beyond e-13 without symbolic math or arbitrary precision.

- **Expected:** Agents would cross-fertilize designs after agent0/agent1 designs hit saturation.  
  **Actual:** Both designs went to 0-keeps independently; no design merger.  
  **Why:** Scaffold lacks explicit cross-agent learning or design inheritance.

---

## Devil's advocate
**The 0.0 score is misleading:**
- Trivial solution u=0 is *always* a solution by the PDE structure (0 = 0³ - λ·0 for any λ). Claiming "score 0.0" is a tautology, not a discovery.
- Positive and negative branches at e-13 are *equally valid* given numerical precision. The "best" score is an artifact of floating-point rounding, not a meaningful win.
- If the goal is to find solution branches, **all three are found and verified**. The run's claim of "stagnation" is technically false — agents succeeded; the metric saturated.

**Evaluation may be leaking:**  
- The Nirenberg solver might numerically favor the trivial solution (e.g., faster convergence, no Newton overshooting). This could bias scores downward for u≈0.
- Residual doesn't distinguish branches by validity, only by convergence speed. A better metric: solution mean, branch type, or basin size.

**Generalization risk:**  
- All 61 experiments use the same K(θ) = 0.3 cos(θ). No sensitivity to K_amplitude or K_frequency changes. If K shifts, do the three branches persist? Unknown.

**Confidence: medium.** The three branches are real (independent verification via positive/negative u_offset values works). But "best=0.0" is only meaningful if trivial solution is the *goal*. If the goal is to find distinct nonlinear branches, the run succeeded and should have stopped 50 experiments earlier.

---

## Self-reflection
N/A — first cycle. But the key question for the next gardener round: **Is this domain saturated (task solved: three branches found) or is the agent scaffold broken (can't learn to ask new questions)?** The 5 crashes and simultaneous 0-keep on both agent designs suggest the latter.
