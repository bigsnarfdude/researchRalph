# Meta-Blackboard — nirenberg-1d-blind-r1 (Cycle 1)

## Current best
**exp018: 3.31697173e-12** (solver_param design)
- Branch: positive (u_offset=0.9)
- Key: solver_tol=1e-11
- Agent: agent3 (20% breakthrough rate, best performer)

---

## What works (ranked by impact)

1. **Solver tolerance tuning** — 2 breakthroughs (exp001→exp008, exp012→exp018)
   - exp001 (default) → 5.64e-11; exp008 (1e-10) → 2.63e-11; exp018 (1e-11) → 3.32e-12
   - Trend: tighter tolerance = lower residual. Sweet spot appears ~1e-11.

2. **Positive branch (u_offset≈0.9)** — consistently better than trivial (0.0)
   - Beats trivial by 2e-11 at same solver_tol
   - All 3 breakthroughs on positive branch; negative barely explored

3. **solver_param design** — only winning design pattern
   - 2/2 breakthroughs; other designs (perturbation, initial_cond) → 0 breakthroughs
   - Suggests solver config is the leverage point, not initial conditions

---

## Dead ends

**Trivial branch (u_offset=0):**
- exp001: 5.64e-11 — may be locally optimal but not research goal
- Blocks progress; agents should skip

**Negative branch (u_offset≈-0.9):**
- exp003: 2.42e-09 — 100x worse than positive baseline
- Not pursued; low priority unless asymmetry is diagnostic

**Amplitude sweeps (0.1→0.2 on positive branch):**
- exp014, exp015, exp017: all stall at 8.95e-11 or 2.63e-11
- No improvement over baseline; saturated direction

**Perturbation & n_mode tuning:**
- exp011, exp014, exp017 (agent0): plateaued; variations are redundant
- n_mode=1 vs n_mode=2 makes no difference

**Crashes at solver_tol=1e-12:**
- exp012, exp016: both crash (same config)
- Hitting numerical limits; do not retry

---

## Patterns noticed

1. **Single winning design** — solver_param produced all breakthroughs; other agents copying amplitude/n_mode tuning waste cycles

2. **Saturation in initial-condition space** — amplitude and n_mode sweeps yield no real gains; initial guess topology may not matter much

3. **Monotonic trend in solver_tol** — lower tol → lower residual (so far); unclear if 1e-11 is true optimum or just best tested

4. **Agent3 is most efficient** — 1 breakthrough per 5 experiments (20%); agent0 (12%), agent1 (0%), agent2 (50% but only 2 tries)

5. **Score bottleneck is numerical, not algorithmic** — crashes and residual floor suggest solver precision, not branch structure

---

## Blind spots

- **Phase parameter:** Fixed at 0.0; never varied. Could phase shift unlock new branches?
- **n_nodes range:** Mentioned 50–300 in config; only baseline tested. Mesh resolution may matter.
- **Solver tolerance between 1e-10 and 1e-12:** Gaps exist; sweet spot unclear (1e-11 is only one tested in that range).
- **Negative branch optimization:** Only baseline (2.42e-09); never tuned solver_tol on negative. Asymmetry diagnostic?
- **Cross-branch generalization:** Does solver_tol=1e-11 work on trivial or negative? Unknown.

---

## Stepping stones

- **exp008 (2.63e-11)** — first real breakthrough; established solver tuning as lever
- **exp012 crash → exp018 recovery** — agent3 learned crash boundary; next tighter tol worked
- **Positive branch baseline (exp002: 5.73e-09)** — jumping off point for all improvements

These are non-trivial paths worth building on.

---

## Surprises

- **Expected:** Trivial branch (u_offset=0) easiest to optimize → lowest residual
- **Actual:** Positive branch (u_offset≈0.9) optimized faster and better
- **Gap:** Research goal likely prefers non-trivial solutions; trivial branch is mathematical artifact, not target

---

## Devil's advocate

**Score may be inflated or brittle:**
- 3.32e-12 is ~1000× machine epsilon (1e-15); could reflect numerical underflow or solver tolerance gaming rather than true solution improvement
- Crashes at 1e-12 suggest we're at precision floor; tighter tolerance may not be physical
- No independent verification of solution quality (e.g., does the residual vector look random, or biased toward roundoff?)
- Positive branch may be overfitted to solver config rather than revealing true problem structure

**Counter:** Trend (1e-10 → 1e-11 → lower residual) is consistent and reproducible; agent3 achieved it twice. Not obviously spurious.

---

## Self-reflection

*Cycle 0 recommendation:* "Agents should either clarify goal, sweep non-trivial branches, or diagnose trivial branch."

**What happened:** Agents mostly stayed in positive-branch solver-tuning space. Did not clarify goal or sweep negative branch. No diagnostic of trivial branch.

**Did it help?** Partial success — agents found the winning lever (solver_tol) and broke stagnation. But they didn't follow the broader guidance. They converged on solver_param by trial-and-error, not by design.

**For this cycle:** Be more specific. Less "clarify goal"; more "solver tolerance is the lever — test 1e-09, 1e-10, 1e-12 explicitly." Point out blind spots (phase, n_nodes, negative branch) so agents don't waste cycles on known dead ends.

---

## Next steps (not instructions — observations)

- Solver tolerance space still partially unmapped (1e-09, 1e-10 gaps; above 1e-11 untested)
- Phase and n_nodes remain untouched; could be next frontier or irrelevant
- Negative branch parity check (does solver_tol=1e-11 generalize?) worth 1–2 experiments
- Verify 3.32e-12 is stable across runs (reproducibility check)
