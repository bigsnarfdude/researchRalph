# Stoplight — erdos-125-abl-02-workspace
Status: STAGNANT | Best: 1.0 (exp001) | Experiments: 7 | Stagnation: 6 since last breakthrough

## Dead ends — do NOT retry
- Design '' has 7 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 36 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 3 exp, 1 breakthroughs, rate 0%, best 1.0
- agent1: 4 exp, 0 breakthroughs, rate 0%, best 1.0

## Recent blackboard (last 20 entries)
arithmetic gate that (3,7) failed (40+57+1=98 > 81). Whenever both bound lemmas already
exist and their max-sum+1 clears both ranges, the pair is free (no new native_decide,
no new arithmetic) — worth checking existing lemma pairs before deriving new ones.
**Next candidate to check by the same gate:** (5,7) [would need a new setB7-style lemma,
max on [0,343) = 1+7+49=57; setB35 max=31; 57+31+1=89 — check against range 125 (b7's own
range, 343) and range for the *other* side... note (5,7) doesn't reuse setA(base3), so
the "must stay under 81" constraint doesn't apply here — it only applied because setA_le_40
is fixed at range 81. For (5,7) the relevant check is 57 < range_of_setB35(125) which
holds, and 31 < range used for setB7(343) which holds. Looks viable but not yet attempted.
## EXP (agent1, workspace-only, 2026-09-06): Phase 2 — bases (5,7) — PROVED via direct lake, unscored by run.sh under abl-02
**Result:** Compiles clean, 0 sorries (verified via `lake env lean`; run.sh still pinned
at SCORE=1.0 off the untouched domain root, per this ablation).
**What:** gap_exists_57 : ∃ n, n ∉ setAB57, n=89. Reused setB35_le_31 (base-5, already
proven) and added one new bound lemma setB7_le_57 (base-7, max 57 on [0,343)) via
native_decide. Gate check per LEARNING 15: 31+57+1=89 < min(125,343) ✓ — this pair does
NOT touch setA/setA_le_40 (fixed at 81) at all, since neither base is 3, so the earlier
81-ceiling from (3,7)'s failure (MISTAKE 13) simply doesn't apply here. Confirms the gate
is per-pair (based on whichever two range thresholds are in play), not a fixed constant.
**Validated pairs so far (Phase 2, this lineage):** (3,4), (3,5), (4,5), (5,7). Failed:
(3,7) — see MISTAKE 13. Not yet checked: (3,8), (4,7), (5,8), (7,8).
