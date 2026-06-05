# Agent7 Learnings

## Progress
- Successfully set up the Erdős #741(ii) proof structure for agent7
- Implemented the main construction definitions: Q, ck, Bk, Fk, Jk, setA, Akn
- Proved helper lemmas: Q_pos, Q_succ, akn_mono
- Implemented gap_lem, which uses rigidity_lem to show that if ck k ∉ T, then Jk k ∩ (T + T) = ∅
- Current status: 5 sorries remaining (elem_from_stage, stage_coverage, basis_lem, rigidity_lem, main theorem body)

## Key Insights
1. **Akn monotonicity**: Akn k ⊆ Akn (k+1) uses the union structure. Since Akn (k+1) = Akn k ∪ {...}, membership in Akn k lifts to the left of the union via Or.inl.

2. **Gap lemma strategy**: gap_lem works by showing that rigidity_lem forces any element in Jk k ∩ (T + T) to have one component equal to ck k. If ck k ∉ T, this is impossible.

3. **Type mismatches with pattern matching**: Trying to destructure `¬ (...)` directly in intro patterns causes issues. Better to intro normally first, then work with the negation separately.

4. **Unfolding Jk and Q**: Jk k = Ico (9 * Q k) (10 * Q k). For k=0: Q 0 = 1, so Jk 0 = [9, 10). This is useful for the base case of inductions.

## What Remains
1. **rigidity_lem**: The crux - proves that elements a, b ∈ setA summing into Jk k must have one be ck k and the other in Bk k. Requires 16-case analysis (4×4 classification).

2. **basis_lem**: Every n ≥ 4 is in setA + setA. Follows from stage_coverage via induction/union.

3. **stage_coverage**: Icc (4 * Q (k+1)) (6 * Q (k+1)) ⊆ Akn (k+1) + Akn (k+1). Requires explicit interval coverage via 8 pair types.

4. **Main theorem body**: Pick large k where Q k > C₁, C₂. Show one partition doesn't have both sumsets syndetic by using gap_lem.

## Technical Notes
- Lean 4 `simp only [Set.mem_singleton_iff]` is useful for singleton sets
- `Set.mem_inter` creates intersection membership from component memberships
- Destructuring `Set.mem_add` via pattern matching can be fragile; better to use rintro or simp + push_neg

## Next Steps (for future agents)
1. Implement elem_from_stage properly using union/iUnion destructuring
2. Complete rigidity_lem with 16-case analysis using classify helper
3. Implement basis_lem using stage_coverage and induction  
4. Complete main theorem with the contradiction argument
