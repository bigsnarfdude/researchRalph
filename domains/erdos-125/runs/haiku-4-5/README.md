model: claude-haiku-4-5-20251001
date: 2026-05-26
result: PROVED (SCORE=1.0, exp001 — first experiment)
commit: pending
generations: 1
agents: 80 (accidental — intended 2, launched 80 due to param order mistake)
max_turns: 80
experiments: 142
proved_count: 114
first_proof_time: 90 seconds (exp001, agent0, 2026-05-26T13:02:18Z)
final_sorry: 0
theorem: gap_exists (∃ n, n ∉ setAB)

Notes:
- Inherited corrected theorem (gap_exists) from Sonnet run overseer intervention
- Inherited fixed run.sh oracle (grep -c || true fix)
- Inherited full blackboard with L1+L2 proofs documented
- 85% of experiments scored SCORE=1.0
- Confirms curation hypothesis: scaffold did the work, not model capability
- Would Haiku succeed under original Sonnet conditions (false theorem, broken oracle)? Likely no.
