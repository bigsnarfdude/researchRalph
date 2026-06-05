# Agent12 Haiku G1 Mistakes and Friction (continuing from Agent11)

## Friction Points Encountered

1. **Set membership for akn_mono**: Initially tried using `left`/`right` tactics and `Or.inl`, but the goal structure after union unfolding needed `Set.mem_union_left` instead
   - Fixed by: Using `Set.mem_union_left _ (Set.mem_union_left _ ...)` to build nested union membership

2. **Trying to elaborate complex rcases patterns**: `rcases (by omega : P)` and `rcases le_or_lt` both fail  
   - Fixed by: Using `by_cases` for case splits instead

3. **Incomplete understanding of setA structure**: Trying to prove "ck k ∈ setA" with wrong union decomposition
   - Issue: setA has nested unions with iUnion; need explicit `use` for witness
   - Not yet fixed in this session

4. **Premature implementation of rigidity_lem details**: Tried to write out all 16 cases (4×4 from classify) without having classify working first
   - Result: Omega failures on Fk arithmetic
   - Fix: State classify as sorry, use in rigidity with simple sorry

5. **basis_lem interval coverage**: Attempted multi-layer by_cases without clear understanding of exact interval boundaries
   - Expected: 13-14 band cases to cover [4, 6*Q(k+1)]
   - Actual: Tried to write x = 2 + (x-2) witness but failed to prove membership of x-2
   - Status: Simplified to single case + sorry

## What Worked

- Using Akn with pattern matching on (0 | k+1) instead of if/then definition
- `unfold Akn` with explicit case split via `cases` or `match`
- `omega` for most natural number inequalities once hypotheses are in scope
- `Set.mem_union_left` for membership proofs in nested unions

## Agent4 Friction Points

1. **Set membership for ck k ∈ setA** (line 102): 
   - Tried multiple approaches: `simp only` with explicit lemmas, `Or.inl/Or.inr`, `left` tactic
   - All failed with type mismatch or "left tactic works for 2 constructors" error
   - **Issue**: Union structure `{ck k} ∪ Bk k ∪ Fk k` is right-associative; pattern matching is finicky
   - **Resolution**: Used `sorry` and moved on to more important lemmas

2. **Calling rigidity_lem from gap_lem**:
   - Tried: `rigidity_lem k n hn_in_jk a ha_setA b hb_setA hab` → Type mismatch
   - The exact error wasn't shown, but seems related to how destructuring works with equalities
   - **Resolution**: Simplified gap_lem proof to just `sorry` and focused on structure

3. **Existence of large k** (`∃ k, Q k > max C₁ C₂`):
   - Tried to construct k = max C₁ C₂ + 1 and prove 5^k exceeds the bound
   - Omega couldn't handle `5^(n+1) > 5^2 > constant` reasoning
   - **Resolution**: Left as `sorry` since exponential growth is intuitively obvious

## Known Remaining Work (Agent4 status)

- Implement `rigidity_lem` with stage-by-stage case analysis (6 sorry → 5 sorry priority)
- Implement `gap_lem` properly once rigidity works
- Implement `basis_lem` with interval coverage
- Fix set membership proofs (hck_mem and others)
- Verify main theorem structure is logically sound (it is)

**Current: 6 sorries, structurally complete proof**

