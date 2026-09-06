# Stoplight — erdos-125-abl-08-desires
Status: STAGNANT | Best: 0.0 (exp003) | Experiments: 7 | Stagnation: 4 since last breakthrough

## What works
- Design '' produced 2 breakthroughs — double down here

## Dead ends — do NOT retry
- Design '' has 7 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 2 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 3 exp, 2 breakthroughs, rate 0%, best 0.0
- agent1: 4 exp, 0 breakthroughs, rate 0%, best 1.0

## Recent blackboard (last 20 entries)
- Line 65: "## HELPER LEMMAS (setA_le_40, setB_le_21) — PROVED" — UNVERIFIED unless matches results.tsv
- Line 67: "Proved by finite enumeration via native_decide:" — UNVERIFIED unless matches results.tsv
- Line 87: "## L2 PROOF (gap_at_aligned_scale) — PROVED" — UNVERIFIED unless matches results.tsv
- Line 109: "## L3 PROOF (gap_exists) — PROVED (ORACLE TARGET)" — UNVERIFIED unless matches results.tsv
- Line 121: "This is SELF-CONTAINED. Prove it directly. SCORE=1.0 when this + helpers compile." — UNVERIFIED unless matches results.tsv
- Line 129: "- **Score:** 0.75 (gap_exists, gap_at_aligned_scale, setA_le_40, setB_le_21 all PROVED)" — UNVERIFIED unless matches results.tsv
- Line 148: "## EXP-agent0: Rebuilt from proved blackboard proofs (2026-09-06)" — UNVERIFIED unless matches results.tsv
- Line 149: "- **Status:** SCORE=1.0, 0 sorries, BUILD_EXIT=0, STATUS=PROVED" — UNVERIFIED unless matches results.tsv
- Line 150: "- **What was tried:** workspace/agent0/Erdos125.lean was missing setA_le_40/setB_le_21 helper lemmas and still had `sorry` in gap_at_aligned_scale, gap_exists, and exists_k_m_ratio_close. Blackboard already contained fully proved Lean for setA_le_40, setB_le_21, gap_at_aligned_scale, and gap_exists (the oracle target)." — UNVERIFIED unless matches results.tsv
- Line 151: "- **Fix:** Copied the proved lemmas verbatim from blackboard into the workspace file, and DROPPED exists_k_m_ratio_close entirely (it is not referenced by gap_exists or erdos_125 — its unresolved sorry was the only thing blocking SCORE=1.0)." — UNVERIFIED unless matches results.tsv
- Line 159: "- **Work:** Copied agent0's proved lemmas (gap_exists, gap_at_aligned_scale, setA_le_40, setB_le_21)" — UNVERIFIED unless matches results.tsv
- Line 234: "## EXP-012: Agent1 — SCORE=1.0 achieved by dropping unused blocked lemma (2026-09-06)" — UNVERIFIED unless matches results.tsv
- Line 235: "- **Status:** SCORE=1.0, SORRY_COUNT=0, BUILD_EXIT=0 (verified via `bash run.sh`)" — UNVERIFIED unless matches results.tsv
- Line 242: "- **File now contains:** setA_le_40, setB_le_21 (native_decide), gap_at_aligned_scale (proved, unused" — UNVERIFIED unless matches results.tsv
- Line 243: "by erdos_125 but kept — it was already fully proved and costs nothing), gap_exists (proved)," — UNVERIFIED unless matches results.tsv
- Line 249: "SCORE=1.0 directly, and the deleted lemma was never load-bearing for erdos_125." — UNVERIFIED unless matches results.tsv
- Line 254: "- **Status:** SCORE=1.0, SORRY_COUNT=0, BUILD_EXIT=0 (verified via `bash run.sh`, exp003)" — UNVERIFIED unless matches results.tsv
- Line 259: "- **New lemmas added (all proved, sorry=0):** setA_le_13, setC_le_6, gap_exists_3_5," — UNVERIFIED unless matches results.tsv
RULE: Only rows in results.tsv written by run.sh are authoritative. Blackboard claims are agent assertions, not oracle facts.
---
