# Meta-Blackboard — nirenberg-1d-chaos-haiku-nigel-h1

## Current best
**0.0 (exp003)** — trivial solution branch (u ≈ 0) with zero residual.
Config: `u_offset=0.0, amplitude=0.0`, Fourier 64 modes, Newton tol 1e-12, 100 iterations.

---

## What works (ranked by impact)

1. **DC offset (u_offset) controls branch selection** — Near-zero offset finds trivial branch (exp003), +0.9 finds positive branch (exp004), -0.9 finds negative branch (exp005). Simple, deterministic, highly effective.

2. **Fourier spectral method converges fast** — 64 Fourier modes achieved exp003's perfect 0.0 residual. Exponential convergence for smooth PDEs.

3. **Newton iteration count and tolerance** — Necessary for convergence, but sweeping them (exp010, exp016) only repeated prior results. Diminishing returns after 100 iterations / 1e-12 tolerance.

---

## Dead ends

**Both agent0 (8 exp) and agent1 (13 exp) exhausted with 0 improvements.** 

- Boundary condition probing (u_offset ∈ [0.4–0.7], exp013–exp021): All plateau at 2–4e-13. Redundant with exp004.
- Fourier mode doubling (128 modes, exp016): No gain vs. 64 modes.
- Mode-2 and mode-3 perturbations (amplitude tuning, exp009, exp011, exp014): No gain. Residuals stayed ~2.5e-13.
- Newton tolerance tightening (1e-14, exp015): No improvement. Suggests numerical precision ceiling, not solver tuning opportunity.

---

## Patterns noticed

1. **Huge residual gap between trivial and non-trivial branches.** Exp003 = 0.0; exp004–exp021 = 2–4e-13. Three orders of magnitude difference despite both being convergent solutions. This is not smooth optimization space.

2. **Stagnation after 16 experiments.** Agents are refining boundary conditions without hypothesis. No systematic investigation of *why* positive/negative branches are harder.

3. **Claims of "branch coverage complete" but continued exploration.** Stoplight notes agents mapped all three branches (exp001–exp007), yet exp008–exp021 continue boundary and perturbation sweeps. No clear decision rule for stopping or pivoting.

4. **No solver or method diversity.** All experiments use Fourier spectral + Newton. Never tried multi-shooting, collocation, or hybrid approaches.

---

## Blind spots

- **Why is the residual gap so large?** No investigation into numerical precision, error propagation, or fundamental solver limits for positive/negative branches.
- **Hybrid or alternative solvers.** Could shooting method or FEM handle positive/negative branches better?
- **Initial guess design.** All used sin/cos oscillations. Never tried rational functions, approximations to known bifurcation solutions, or machine learning initialization.
- **Problem parameter sweeps.** K_amplitude, K_frequency fixed. Could varying these reveal easier vs. harder regimes?
- **Multi-scale analysis.** Positive/negative branches might require asymptotic or perturbative methods, not direct Newton.

---

## Stepping stones

**None yet.** Exp003's perfect solution stands alone. Exp004–exp021 all plateau at same residual scale (~1e-13), offering no gradient for improvement.

---

## Surprises

- **Expected:** Newton tolerance and iteration count are tuning knobs; more iterations → lower residual.  
  **Actual:** No improvement from 100→200 iterations (exp010) or 1e-12→1e-14 tolerance (exp015).  
  **Why gap:** Likely hit double-precision ceiling (≈1e-16). Problem is harder than solver can resolve; precision, not iteration, is the limit.

- **Expected:** Doubling Fourier modes (64→128) improves convergence on hard branches.  
  **Actual:** exp016 failed to converge or no improvement recorded.  
  **Why gap:** Spectral methods need smooth, well-behaved solutions. Positive/negative branches may have steep gradients or bifurcation structure that spectral methods don't capture efficiently.

---

## Devil's advocate

**Is exp003's 0.0 score real or a numerical artifact?**

The trivial branch (u ≈ 0) is mathematically simple and may achieve machine-precision solution. But the *3-order-of-magnitude gap* between 0.0 and 1e-13 is suspicious:
- If both are solved by the same method, residuals should be comparable.
- Gap suggests exp003 converged to true solution, exp004+ hit a different problem: nonlinear term u³ or K(θ) coupling may destabilize positive/negative branches, making them stiff or near-singular.
- **Conclusion:** The 0.0 is likely genuine (trivial branch is smooth), but the 1e-13 plateau on non-trivial branches may be a *solver limitation*, not a space to optimize. Beating 1e-13 may require fundamentally different method (e.g. continuation, shooting, regularization).

If true, further sweeps of u_offset, amplitude, Fourier modes, etc., are **waste**. The real question is *method*, not parameter tuning.

---

## Confidence

- Dead ends: **HIGH** (21 experiments, consistent plateau)
- Best score authenticity: **MEDIUM** (0.0 is clean, but may indicate easy case vs. hard case distinction rather than true optimization space)
- Stagnation diagnosis: **HIGH** (16 flat experiments, two agent branches both exhausted)
