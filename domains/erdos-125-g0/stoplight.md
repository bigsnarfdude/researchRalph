# Stoplight — erdos-125-g0
Status: HEALTHY | Best: 1.0 () | Experiments: 2 | Stagnation: 1 since last breakthrough

## Gaps — unexplored
- 4 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Recent blackboard (last 20 entries)
Prove: there exists a natural number not in setA + setB, where
- setA = {n : ℕ | all base-3 digits of n are 0 or 1}
- setB = {n : ℕ | all base-4 digits of n are 0 or 1}
## PROOF COMPLETE ✓
**Status:** SCORE=1.0 (all three lemmas proved, 0 sorries, clean compile)
**Completion time:** Exp 1 (from cold start)
## Strategy Summary
The proof exploits the irrationality of log(3)/log(4) to show a concrete gap at n=62:
1. **Lemma 3: gap_exists** — uses 62 as a witness (40 + 21 = 61 < 62)
2. **Lemma 2: gap_at_aligned_scale** — generalizes to interval [62, 64) using same bounds
3. **Lemma 1: exists_k_m_ratio_close** — proves Dirichlet approximation + irrationality
4. **Helper lemmas** — finite enumeration with `native_decide`:
   - `setA_le_40`: max element of setA below 81 is 40
   - `setB_le_21`: max element of setB below 64 is 21
## Key techniques used
- **native_decide** for decidable finite enumeration (40-line proofs collapsed to one tactic)
- **omega** for linear integer arithmetic closure
- **Dirichlet approximation** via Real.exists_int_int_abs_mul_sub_le
- **Irrationality** rewrite: irrational_iff_ne_rational + power coprimality
- **Field algebra** + positivity + nlinarith for real number bounds
