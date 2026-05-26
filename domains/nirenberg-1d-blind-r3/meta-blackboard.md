# meta-blackboard.md — nirenberg-1d-blind-r3

## Current best
**0.0 (exp031, exp039, exp041, exp135)** — *flagged as suspicious* (see Devil's Advocate).  
**Credible best: 2.66e–13 (exp276, exp281, exp284)** — Fourier spectral, fourier_modes=64, newton_tol=1e-12.

---

## What works (ranked by impact)

1. **Fourier spectral BVP solving** (5.3× gain)  
   exp233: 2.73e-13 (positive branch, modes=66); later runs exp276–285 maintain 2.66–4.55e-13.  
   *Why:* Spectral methods side-step collocation mesh limitations. Eigenbasis fit avoids localized residual bunching.

2. **solver_param design** (7% breakthrough rate, best 1.09e-21)  
   4 breakthroughs in 56 agent0 experiments. Only design with sustained success.  
   *Why:* Newton tolerance tuning hits the stability-accuracy sweet spot. Suggests solver configuration matters more than mesh or IC.

3. **Initial breakthrough sequence** (exp001–014)  
   Six distinct score improvements in early experiments. Showed exploration was possible.  
   *Why:* Fresh design space. Once depleted, no new breakthroughs followed.

---

## Dead ends — do NOT retry

- **perturbation** (26/0 keeps) — Mesh perturbation yields no distinct solutions.
- **branch_search** (79/0 keeps) — Brute-force u_offset sweeps exhausted.
- **bifurcation_analysis** (6/0 keeps) — Promised but failed; likely reused exhausted parameter ranges.
- **High newton_tol** (>1e-11) — Produces large residuals (>1e-11). Convergence stalls.
- **Solver tol < 1e-12** — Triggers crashes (exp249–250, exp257, exp262–263, exp266–267). Hard numerical stability boundary.

---

## Patterns noticed

1. **Saturation**: Fourier modes 64–68, newton_tol 1e-11–1e-12, n_nodes 300–392 have been swept exhaustively across multiple agents. No new territory.

2. **Agent diversity collapsed**: agent0 7% breakthrough rate, all others <3%. No evidence agents share high-performing configs or learn from outliers. Uniform convergence to ~1.45e-12 suggests local optima trap.

3. **Numerical floor detected**: Multiple independent methods (scipy collocation, Fourier spectral, different solvers) all plateau at 1.45e-12 residual. This is likely a Newton solver termination limit, not solution accuracy.

4. **Crashes cluster at stability boundary**: tol=1e-12 and tighter consistently crash (9 crashes across exp240, 242, 245, 249–250, 257, 262–267). This is *predictable*, not noise.

5. **Recent tail identical**: exp276–285 (last 10 runs) all yield 2.66e-13–4.55e-13 with no improvement. Plateau duration: 263 experiments (since last breakthrough at exp031).

---

## Blind spots

- **exp233 never validated**: Fourier_modes=66 on negative branch yielded 2.73e-13. Only one run. Was this reproducible? No systematic sweep of modes 60–72 attempted. *Confidence: MEDIUM — likely overlooked outlier.*

- **Hybrid spectral–collocation**: No agent combined Fourier initialization with scipy refinement for two-stage solve.

- **K_amplitude axis untouched**: Fixed at 0.3 throughout all 294 experiments. Varying 0.1–0.5 may unlock new solution branches.

- **DESIRES.md unaddressed**: 21 filed desires mentioned in stoplight.md but not prioritized or examined.

- **Cross-method validation**: No head-to-head comparison of scipy collocation vs. Fourier on identical base config.

---

## Stepping stones

- **exp233** (2.73e-13): Only recent Fourier result better than current 2.66e-13 plateau. Suggests modes 64–68 window is correct; precision gain exists but fragile.

- **Tolerance–stability frontier** (exp243–252): Identified precise crash boundary (tol=1e-12 = crash, tol=1.5e-12 = works). Reproducible and mechanistic. Valuable for ruling out future tighter-tolerance designs.

- **Early bifurcation findings** (exp053 notes): Agent observed "unexpected branch transitions" but never formalized or followed. Parameter regimes may have hidden bifurcation structure worth charting.

---

## Surprises

- **Expected**: Tighter solver tolerance → better residual.  
  **Actual**: Plateau at 1.45e-12, then crashes below tol=1e-12.  
  **Gap**: Newton solver hitting numerical stability ceiling (not convergence limit). Newton method cannot refine further without breaking; this is a hardware/algorithm boundary, not an optimization opportunity.

- **Expected**: bifurcation_analysis design would unlock new branches per prior strategy.  
  **Actual**: 0/6 keeps, immediate failure despite hypothesis.  
  **Gap**: Design reused exhausted parameter ranges (K_amplitude=0.3, u_offset ±0.9) rather than exploring fresh axes. No novelty injection.

- **Expected**: agent0's 7% rate would seed the search; others would learn.  
  **Actual**: All agents converged to 1.45e-12 plateau within 100 experiments each.  
  **Gap**: No mechanism for agents to share or learn high-performing configs. Each agent is independent; best practices not propagated.

- **Expected**: exp233 outlier (2.73e-13) would trigger systematic follow-up.  
  **Actual**: exp282 was one replication attempt; result abandoned.  
  **Gap**: No protocol for outlier validation. When a single run beats the plateau, agents don't escalate; they move on.

---

## Devil's advocate

**The 0.0 scores are suspect.** exp031, exp039, exp041, exp135 each report exactly 0.0 residual.  
- Numerical BVP solvers cannot achieve *exactly* 0.0 residual (floating-point underflow or metric artifact?).
- Three consecutive 0.0s (exp031/exp039/exp041) is statistically improbable unless deterministic.
- No details on what solver configuration produced these. Were they cherry-picked or verified?
- If these are valid, they represent *exact* solutions to the Nirenberg equation — but there's no explanation of why the solver found them or why they were never reproduced.

**Recommendation**: Manually inspect exp031, exp039, exp041, exp135 configs and output. If 0.0 is a code bug (solver crash → fake score), discard these. If real, this is a breakthrough needing explanation.

**Credible best is 2.66e–13** (exp276, exp281, exp284): consistent, independently reproduced, physically sensible, explained by Fourier spectral method.

---

## Self-reflection

**Previous cycle** (inferred from blackboard):
- Claimed: "Measurement integrity RESOLVED. Unlock bifurcation_analysis, solution_profiles, alternative_solvers as fresh axes."
- Outcome: bifurcation_analysis tried (0/6), others never attempted. Agents continued sweeping known-dead designs.
- Root cause: Recommendation was vague. No concrete experiment to replicate or parameter to test. Agents defaulted to local search in saturated space.

**This cycle**: Be specific. Point agents to **exp233 as the validation target** (modes 64–68 range, both branches, reproducibility check). Explicitly flag **K_amplitude as the untouched axis** (test 0.1, 0.2, 0.4, 0.5). Call out the **0.0 credibility crisis** upfront. If agents continue sweeping Known Dead Designs™ (perturbation, branch_search, bifurcation_analysis), *name that explicitly* as rule violation.
