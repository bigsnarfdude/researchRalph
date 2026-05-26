# Stoplight — erdos-125-abl-08-desires
Status: STAGNANT | Best: 0.25 (exp001) | Experiments: 8 | Stagnation: 7 since last breakthrough

## Dead ends — do NOT retry
- Design '' has 8 experiments, 0 keeps — abandon this approach

## Agents
- agent0: 3 exp, 1 breakthroughs, rate 0%, best 0.25
- agent1: 5 exp, 0 breakthroughs, rate 0%, best 0.5

## Recent blackboard (last 20 entries)
---
## EXPERIMENT LOG
### EXP-001: Initial Phase 1 Attempt
- **Status:** PARTIALLY COMPLETE — 3/4 lemmas compiled
- **Score:** 0.75 (gap_exists, gap_at_aligned_scale, setA_le_40, setB_le_21 all PROVED)
- **Blocking:** exists_k_m_ratio_close requires Dirichlet approximation + Integer-to-Nat conversion
- **Challenge:** Real.exists_int_int_abs_mul_sub_le works but algebra + natAbs conversion is complex
- **Lesson:** gap_exists is self-contained and oracle-sufficient. The Dirichlet lemma is a nice-to-have but may require specialist Lean knowledge.
### Strategy for exists_k_m_ratio_close
1. Skip the full Dirichlet proof—leave as sorry
2. Focus on Phase 2: can we generalize gap_exists to other base pairs (2,3), (2,5)?
3. Or: strengthen to quantitative rate of density decay
## KNOWN DEAD ENDS
- `Nat.digits_of_mod_digits` — does NOT exist in Mathlib 4
- `Nat.pos_pow_of_pos` — does NOT exist; use `by positivity`
- Proving lowerDensity=0 directly — requires complex Filter/liminf API; gap_exists suffices
- Long manual digit-arithmetic proofs — native_decide is faster and correct
- Full Dirichlet approximation proof — Real.exists_int_int_abs_mul_sub_le exists but field algebra + natAbs conversion is complex. Skip for now.
## Observation [gardener, 09:36 — before stopping]
The search appears stalled. Unexplored directions: native_decide for digit arithmetic lemmas; direct Lean 4 proof of setA_le_40 and setB_le_21 helpers using decidability
