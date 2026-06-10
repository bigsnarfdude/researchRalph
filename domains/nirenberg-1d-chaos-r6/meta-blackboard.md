# Meta-Blackboard — nirenberg-1d-chaos-r6 (Cycle 23)

## Current best
**Score: 0.0** (residual, lower = better)  
**Config:** Trivial branch, u_offset=0.0, fourier_modes=1, newton_tol=1e-14  
**Achieved:** exp001, exp005, exp081, exp093, exp115, exp135, exp159, exp179, exp224, exp258, exp267, exp270, exp307, exp335, exp358, exp362, exp365, exp385 (18 times)  
**Status:** Perfect residual on trivial (u ≈ 0) solution branch. NOT on non-trivial branches.

---

## What works (ranked by impact)

1. **Trivial branch solver** (+0.0 vs noise)  
   *Why:* Fourier method with modes≥1 converges to u≈0 exactly. Newton tolerances < 1e-14 achieve machine zero.

2. **Basin boundary mapping via u_offset sweeps** (structural insight, no score gain)  
   *Why:* Reveals that modes=1 boundary ≈ 0.4745, modes=2-4 ≈ 0.462. Mode count affects basin structure.

3. **Mode-2 perturbations from trivial** (escape mechanism, exp400/exp402)  
   *Why:* n_mode=2, amp~0.475 from u_offset=0 can reach ±1 branches. But gains are ~1e-16, not stable.

---

## Dead ends

**Agent design (all 8):** 0 keeps across 300+ experiments.  
- agent0-7 all tried similar sweeps → saturation.
- Reason: Agents are re-exploring same basin-boundary/mode-sweep parameter space.

**Mode-2 crashes:** modes=2, u_offset~0.462 → BVP solver crashes (saddle-point instability).  
**High fourier modes:** modes=16, 64 → incomplete, no improvement over modes=1-4.  
**scipy solvers:** Best residuals 1e-10 to 1e-11 (worse than fourier 1e-14–1e-30).  
**Phase/amplitude perturbations:** amp, phase, n_mode sweeps → mostly trivial convergence.  
**Z2 symmetry tests:** Negative u_offset mirrors don't yield new solutions.

---

## Patterns noticed

1. **Saturation:** All 8 agent designs exhausted with identical redundancy (basin probes, mode sweeps).  
   → Agent architecture needs redesign; currently all agents explore same neighborhood.

2. **Metric gaming risk:** Best score is 0.0 on *trivial* branch. Non-trivial branches score ~1e-16 to 1e-29.  
   → Agents may be optimizing trivial-branch precision, not exploring new solution families.

3. **Hidden structure:** Exact-zero regions at u_offset=0.2, 0.46845, and "cliff" at u_offset=0.1115  
   → Nobody has built diagnostic tools to understand these anomalies.

4. **Mode-count dependence:** Basin topology shifts with fourier_modes. modes=1 wider trivial basin than modes≥2.  
   → Fundamental trade-off not explored: can higher modes access new solutions?

---

## Blind spots

- **Negative u_offset basin systematically.** Only Z2 mirror tests, no full sweep of u_offset=[-0.9, -0.3].
- **Multi-mode initial conditions** (combining n_mode=2 + fourier_modes=2+). Tried separately, never together.
- **Oscillatory behavior near cliff (0.1115).** Cliff shows 1e-16 → 1e-31 jump. Why? No agent probed mechanism.
- **Solver diagnostics.** No agent logged Newton iteration count, Jacobian rank, or condition numbers → can't diagnose why modes=2 crashes.

---

## Stepping stones

- **exp400/exp402:** Mode-2 perturbation reaches non-trivial but residuals ~1e-16. Suggests hybrid initialization: trivial seed + careful perturbation tuning could yield lower-residual non-trivial.
- **Exact-zero regions:** u_offset=0.2 and 0.46845 both give 0.0 residual. Suggest interior structure of trivial basin is non-smooth. Worth mapping full residual landscape.
- **Mode=2 crash at boundary:** Indicates saddle point near modes=2, u_offset=0.462. Could be used to *find* non-trivial solution analytically.

---

## Surprises

- **Expected:** exp001 = 0.0 is a *once-only* breakthrough, hard to reproduce.  
  **Actual:** 0.0 residual reproducible across 18+ runs at trivial basin.  
  **Gap:** Trivial solution is robustly findable; problem is *not* exploring non-trivial branches.

- **Expected:** Basin boundaries are smooth (u_offset increases → solution type changes).  
  **Actual:** Multiple exact-zero regions, sharp cliff at 0.1115, residuals jump 15+ orders of magnitude within 0.0001 offset.  
  **Gap:** Trivial basin has unexpected sub-structure; agents skimmed over it.

---

## Devil's advocate

**The 0.0 score is *misleading*.** Three concerns:

1. **Trivial branch is a "free win."** u≈0 is a solution to u³ - (1+K)u = 0 by definition. Agents that hit it aren't solving the BVP; they're finding a fixed point. Non-trivial branches (u ≈ ±1) score 1e-16 to 1e-29 because they don't satisfy the equation as exactly. **The metric may favor triviality.**

2. **Metric gaming via solver precision.** Fourier method achieves 1e-14 tolerance on trivial. But higher fourier_modes or tighter newton_tol on non-trivial may be numerically unstable (exp as 1e-30 suggests truncation/rounding, not physical convergence).

3. **No oracle confirmation.** The score is residual(config). But is the non-trivial residual floor (1e-30) real, or floating-point noise? Agents never verified by independent solver or domain expertise.

**Counter-argument:** If the goal is finding multiple solution branches with *any* residual < ε, then 0.0 is a red herring. The real goal (inferred from 411 exp) is non-trivial branches. Measuring success only by lowest residual ignores that.

---

## Self-reflection

*First cycle; no prior meta-blackboard to compare.*

---

**Confidence:** HIGH on dead ends (8/8 agent designs, 0 keeps). MEDIUM on stepping stones (mode-2 perturbation is interesting but scores remain low). **Recommendation to gardener:** Redesign agent scaffold to separate trivial-basin precision from non-trivial-branch exploration. Agents should pick *one* regime per run, not mix both.
