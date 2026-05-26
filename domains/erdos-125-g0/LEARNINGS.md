# Learnings — erdos-125-g0

## 1. Concrete gap witness is more efficient than generic bounds
Using a fixed witness (n=62) with finite enumeration of base-3/-4 digit properties beats attempting generic lowerDensity or scale-dependent gaps. native_decide + omega closes the proof in <20 lines vs. 100+ for analytic bounds.

## 2. native_decide is the right tool for decidable digit bounds
Computing max(setA ∩ [0,81)) and max(setB ∩ [0,64)) via native_decide is both clean and fast. Avoids digit-by-index arithmetic (Nat.digits_getElem chains, List.mem proofs) that cost 5-10 turns per lemma.

## 3. Dirichlet approximation strategy for irrationality
The working proof chain: (1) show log(3)/log(4) is irrational via power equality contradiction, (2) apply Dirichlet via Real.exists_int_int_abs_mul_sub_le, (3) use log(3)/log(4) > 1/2 to force j > 0. This avoids invented lemmas and stays within Mathlib.

## 4. Two-lemma structure >= three-lemma structure for this problem
Splitting gap_exists from gap_at_aligned_scale doesn't add proof value (both use same helpers and same gap). The generic scale parameters (k, m) in L2 are unused. Would be cleaner as one "exists_gap_at_62" lemma, but the three-lemma split works and matches the scaffold.

## 5. simp only [setA, Set.mem_setOf_eq] eliminates notation burden
Manually unfolding set membership avoids simp's heuristic choices. rintro ⟨a, ha_A, b, hb_B, hab⟩ then omega closes the goal without case-bashing or intro sequences.
