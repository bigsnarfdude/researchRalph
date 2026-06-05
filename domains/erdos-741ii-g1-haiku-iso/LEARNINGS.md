# Agent4+Agent11 Haiku G1 Learnings

## Architecture That Works (from prior Opus agents)

1. **Definitions**: Q k = 5^k, ck k = 4*Q k, Bk k = [5*Q k, 6*Q k - 1], Fk k = [10*Q k - 1, 15*Q k], Jk k = [9*Q k, 10*Q k)
   
2. **Key lemmas** (in order of implementation difficulty):
   - `akn_mono`: Akn k ⊆ Akn (k+1) — Use `Set.mem_union_left` to lift to unions
   - `classify`: Any e < 10*Q k is either ≤ 3*Q k (stage < k), = ck k, ∈ Bk k, or ∈ Fk k
   - `rigidity_lem`: For n ∈ Jk k, any sum a + b = n requires (a = ck k ∧ b ∈ Bk k) or (b = ck k ∧ a ∈ Bk k)
   - `gap_lem`: If ck k ∉ T, then Jk k ∩ (T + T) = ∅ (follows from rigidity)
   - `basis_lem`: Every n ∈ [4, 6*Q(k+1)] is a sum of two elements from Akn(k+1)

3. **Main theorem structure**:
   - Use setA = {2, 3} ∪ ⋃_k ({ck k} ∪ Bk k ∪ Fk k)
   - For partition A₁ ⊔ A₂, pick k = C₁ + C₂ + 1
   - ck k ∈ A₁ or A₂; WLOG say A₁
   - Then ck k ∉ A₂, so by gap_lem: Jk k ∩ (A₂ + A₂) = ∅
   - But A₂ + A₂ syndetic means ∃ m ∈ [9*Q k, 9*Q k + C₂] ⊆ Jk k
   - Contradiction!

## Critical Lean Tactics

- **Don't use** `rcases le_or_lt x y` — it elaborates to metavariable. Use `by_cases h : x ≤ y` instead
- **Use** `by omega` for natural number arithmetic, NOT `linarith`
- **For set membership**, after `simp only [mem_Icc]`, omega can handle bounds
- **For unions**, use `Set.mem_union_left / Set.mem_union_right` or `Or.inl / Or.inr`
- **For singleton sets**, after `simp only [mem_singleton_iff]`, the goal becomes `x = y`

## Friction Points

1. **setA membership**: It's a complex nested union. Construction proofs like "ck k ∈ setA" need explicit unfolding and `use` tactics
2. **Bk/Fk arithmetic**: Natural number subtraction in `Icc (10*Q k - 1) (15*Q k)` requires omega, not linarith
3. **akn_mono** with nested `Akn` definition: Need `Set.mem_union_left` not `left` tactic (different contexts)
4. **Interval union coverage** (basis): 13-14 band by_cases is tedious but mechanical — parameterize via `cover_pairs` helper

## Next Agent Strategy (Agent4 plan - executed through Agent11)

1. Implement `akn_mono` fully using `Set.mem_union_left` pattern
2. State but don't prove `classify` (just use sorry)
3. Use `classify` in `rigidity_lem` with a 4×4 rcases, let omega handle impossible combos
4. Implement `gap_lem` as simple contradiction from rigidity
5. Implement `basis_lem` via `cover_pairs` helper with by_cases ladder
6. Complete main theorem using gap_lem + large k selection

## Agent12 Progress

Successfully structured complete proof skeleton with 14 sorries:
- All definitions correct (Q, ck, Bk, Fk, Jk, setA, Akn, IsSyndetic)
- All key lemmas stated in logically correct order
- Main theorem structure complete with basis and partition parts
- Helper lemmas for k selection and membership included

**Critical friction point discovered**: Union type destructuring in Lean 4
- `left`/`right` tactics fail on `A ∪ B ∪ C` style unions after unfolding
- Solution: Use `Set.mem_union_left / Right` explicitly or `obtain` with pattern matching
- Affected: akn_mono, ck_mem_setA, akn_subset_setA

**File compiles successfully** with SCORE=0.0 (14 sorries remaining)

**Priority for next agent**: Implement basis_lem (13-14 by_cases for interval bands)
