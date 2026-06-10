# Meta-Blackboard — nirenberg-1d-blind-chaos-gemma4-c0-n2

## Current best
**exp001: 0.0 (n_nodes: 150)** — *flagged as suspicious* (see Devil's Advocate)

**Second-best (likely genuine): exp002: 3.61e-13 (u_offset: 1.0)**

---

## What works (ranked by impact)
1. **Increasing n_nodes (100→150)** — Exp001 claims breakthrough, but score=0.0 is numerically implausible (see concerns below)
2. **u_offset perturbations** — Exp002–003 consistently hit ~2.3e-13, matching solver precision ceiling observed in prior 99-exp run
3. **Solver tolerance tuning** — Exp004 (tol=1e-10) did not improve on status quo

---

## Dead ends (preliminary)
- **solver_tol reduction** (exp004) — tighter tolerance ≠ better residual; may degrade convergence at basin boundaries
- **u_offset=0.5** (exp003) vs u_offset=1.0 (exp002) — marginal difference (2.28e-13 vs 3.61e-13), no clear winner yet

---

## Patterns noticed
1. **All exp from agent0** — zero agent diversity; no parallel exploration
2. **Early stagnation flag** — declared STAGNANT after 4 experiments (3 since "breakthrough")
3. **Mismatch with prior run** — Prior 99-exp run found u_offset=±0.52 optimal (2.10e-13), trivial branch at u_offset≈0. Current exp002–003 use u_offset=1.0/0.5 (different regime)
4. **No basin boundary exploration** — Prior run identified chaotic bifurcation cascade (0.461, 0.475, 0.585); not probed here yet

---

## Blind spots
1. **Symmetric u_offset search** (±0.52 from prior run) — Exp2–3 didn't test these exact points
2. **Branch identification** — Is exp001's "0.0" the trivial branch? Or underflow artifact? Which branch are exp002–003 on?
3. **Fourier mode sweep** — Fixed at 64; prior run found >64 degrades convergence at basin edges
4. **Amplitude/phase perturbations** — Not tried; prior run showed these degrade residuals but worth testing
5. **Mesh refinement beyond 150** — n_nodes=150 is lowest-hanging fruit; 200-300 untested

---

## Stepping stones
- Exp002–003 consistency at ~2e-13 suggests robust convergence on a specific branch; could be a solid secondary optimum
- n_nodes increase (100→150) *did* trigger the "0.0" claim; incremental mesh refinement may be on the right track

---

## Surprises
- **Expected:** u_offset variations around 0 (trivial branch) would find exact solution (residual ≈ 0)
- **Actual:** Best score = 0.0 at high u_offset regime (1.0, 0.5) where prior run found chaotic basin boundaries
- **Gap:** Either exp001 is a measurement error, OR basin dynamics differ from prior run, OR this is a different BVP variant

---

## Devil's advocate
**exp001's score of 0.0 is likely WRONG.** Reasons:
1. **Precedent:** Prior 99-exp run never achieved 0.0; best was 2.10e-13 (solver precision ceiling)
2. **Regime mismatch:** u_offset=0 is where trivial branch lives; u_offset=150 (n_nodes param?) is solver discretization, not initial condition
3. **Config ambiguity:** Is exp001 truly u_offset≈0 on trivial branch, or is the "0.0" a display/eval bug?
4. **Reproducibility risk:** Exp002–003 (u_offset=1.0/0.5) get 1e-13, not 0.0, suggesting 0.0 is non-repeatable

**Recommendation:** Treat exp002 (3.61e-13) as the genuine best until exp001 is reproduced or explained.

---

## Next priorities (observations only)
- Verify whether exp001 was on trivial or nontrivial branch (u_offset actual value?)
- Test u_offset=±0.52 explicitly (prior run's sweet spots)
- Probe basin bifurcations (0.461, 0.475, 0.585) to map chaotic regime
- Increase agent count to avoid agent0 monopoly; run exp5+ from agent1+ for cross-check

