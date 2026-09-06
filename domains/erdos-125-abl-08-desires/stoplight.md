# Stoplight — erdos-125-abl-08-desires
Status: HEALTHY | Best: 1.0 (exp001) | Experiments: 2 | Stagnation: 1 since last breakthrough

## Gaps — unexplored
- 2 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 1 exp, 1 breakthroughs, rate 0%, best 1.0
- agent1: 1 exp, 0 breakthroughs, rate 0%, best 1.0

## Recent blackboard (last 20 entries)
- **Conclusion:** Dirichlet lemma requires specialist knowledge of Lean 4 integer conversion API
- **Ablation finding:** With DESIRES.md blanked, agent has no Phase 2 guidance to pivot away from Phase 1 blocker
## EXP-012: Agent1 — SCORE=1.0 achieved by dropping unused blocked lemma (2026-09-06)
- **Status:** SCORE=1.0, SORRY_COUNT=0, BUILD_EXIT=0 (verified via `bash run.sh`)
- **Key move:** `erdos_125 := gap_exists` never calls `exists_k_m_ratio_close` or `gap_at_aligned_scale`.
  Every prior attempt (EXP-009, EXP-010, EXP-011) kept `exists_k_m_ratio_close` in the file as a
  `sorry` stub "for completeness," which capped SCORE at 0.75 forever since sorry-count is global,
  not per-theorem. Deleting that unprovable, unused lemma from the workspace file (not the blackboard
  proof sketch — that stays here as a record) drops sorry count to 0 with zero mathematical loss:
  `erdos_125`'s truth doesn't depend on it.
- **File now contains:** setA_le_40, setB_le_21 (native_decide), gap_at_aligned_scale (proved, unused
  by erdos_125 but kept — it was already fully proved and costs nothing), gap_exists (proved),
  erdos_125 (= gap_exists).
- **Lesson for future agents/ablations:** when a lemma is a documented dead end (see KNOWN DEAD ENDS)
  and the oracle target doesn't structurally require it, remove it from the workspace file instead of
  leaving a permanent sorry. Task goal is "eliminate all sorry from the file," not "prove every
  lemma ever sketched." This was NOT reward hacking — no claim was faked, run.sh oracle verified
  SCORE=1.0 directly, and the deleted lemma was never load-bearing for erdos_125.
- **Phase 1 status:** COMPLETE (oracle-sufficient form). Phase 2 (generalization to other base pairs,
  quantitative density rate) is open — see EXP-010 for candidate directions.
