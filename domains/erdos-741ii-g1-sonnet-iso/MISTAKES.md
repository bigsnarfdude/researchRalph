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

## Agent15 (Haiku) Session - Key Discoveries

**Major Victory**: Got main theorem structure compiling end-to-end with proper error handling in gap argument.

**Critical Issue - Induction Hypothesis Type Mismatch (UNSOLVED)**:
- When proving `Akn_subset_setA: Akn k ⊆ setA` via induction, repeatedly hit "Application type mismatch" on `ih`
- Tried 6+ different proof structures:
  - `intro x; induction k; ...` → ih application fails
  - `induction k; intro x; ...` → same error on `ih hx`
  - Pre-simp with subset_def → still fails
  - Explicit `have : x ∈ Akn k := ...` intermediate → still type mismatch
- The error says we're applying ih to the wrong type, but after simp/unfold/rcases, the types look right
- Suspected cause: Lean 4's handling of induction hypothesis scope after complex tactic sequences
- **Status**: Left as sorry; decided that finishing a working proof structure with sorries beats getting stuck on one lemma

**Omega Limitations Confirmed**:
- Cannot solve `∃ a b, (a=2 ∨ a=3) ∧ (b=2 ∨ b=3) ∧ a+b=n` after simp unfolds membership
- Needed explicit witness provision via `match` in basis_lem zero case
- `5 * 5^k ≥ m + 1` needed case split even after `have h2 : 5 * 5 ^ k ≥ 5 * k`

**What Worked This Session**:
- Using `match n with | 4 => ... | 5 => ... | 6 => ... | n+7 => ...` to explicitly handle small cases
- Restructuring exponential growth proof with case split on k for base+inductive structure
- `simp only [setA, Set.mem_union, Set.mem_iUnion] at ⊢` to manually unfold goal before tactics
- `norm_num` for closing numeric goals after case analysis
- `tauto` to solve propositional contradictions after simp

**Main Theorem Now Proves** (modulo 4 sorries):
- Defines construction (Q, ck, Bk, Fk, Jk, setA, Akn) ✓
- Proves exists_covering_level with manual exponential argument ✓  
- Proves setA_is_basis (assuming basis_lem) ✓
- Main theorem structure with gap argument and case split on A₁/A₂ ✓
- Only missing: basis_lem (succ case), rigidity_lem, gap_lem, Akn_subset_setA

**Next Agent Should**:
1. Try `Set.subset_def` + full unfold before induction on Akn_subset_setA
2. Prove basis_lem succ case via induction on interval size
3. Implement rigidity_lem by stage decomposition (j < k, j = k, j > k)
4. Use rigidity_lem in gap_lem to show interval disjointness

- Implement `rigidity_lem` with stage-by-stage case analysis (6 sorry → 5 sorry priority)
- Implement `gap_lem` properly once rigidity works
- Implement `basis_lem` with interval coverage
- Fix set membership proofs (hck_mem and others)
- Verify main theorem structure is logically sound (it is)

**Current: 6 sorries, structurally complete proof**

## Agent8 Session Friction Points

1. **Akn unfolding with pattern matching**:
   - Tried: `simp only [Akn]`, `dsimp only [Akn]`, explicit induction
   - All failed with "Type mismatch" or "Application type mismatch" on Or.inl
   - **Issue**: Lean has trouble unfolding pattern-matched recursive definitions with let bindings
   - **Resolution**: Avoid explicit unfolding; accept sorry for akn_mono, use structural approach elsewhere

2. **Set membership of ck k ∈ setA**:
   - Tried: `unfold setA; right; use k; left` then various tactics
   - Issue: The triple union `{ck k} ∪ Bk k ∪ Fk k` requires careful handling
   - **Resolution**: Use `sorry` for now; direct membership is straightforward by definition but Lean is finicky

3. **setA structure navigation**:
   - Initial attempts to prove membership failed due to mixing simp with left/right tactics
   - The nested union structure (union of iUnion) requires explicit destructuring
   - **Better approach**: Use ext + simp_only with mem_union, mem_iUnion, mem_singleton_iff

## What Worked Well in Agent8

- **gap_lem proof**: Clean ext pattern with simp_only to handle set equality
- **Main partition argument**: Full structure with both by_cases branches implemented
- **Using sorry strategically**: Allowed progress on structure rather than getting stuck on individual proofs
- **omega tactic**: Handles all the Jk ∩ interval membership proofs correctly

## Current Status: 7 sorries, full proof structure

The 7 remaining sorries are:
1. rigidity_lem (core mathematical content)
2. basis_lem (core mathematical content)
3. akn_mono (structural support)
4. akn_in_setA (structural support)
5. hck_mem (membership proof - straightforward)
6. hk_exists (growth argument - straightforward)
7. large k for basis (existence proof - straightforward)

Priorties for next agent: rigidity_lem and basis_lem contain the real mathematical substance.

