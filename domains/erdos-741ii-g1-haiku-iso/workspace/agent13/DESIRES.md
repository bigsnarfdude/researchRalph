# Agent13 Desires

## What Would Help Most

1. **Better tactic for Or constructors on unions** — currently broken for nested unions
   - Pattern like `exact_or_left` or `left_in_union` that handles A ∪ B ∪ C ∪ D
   - Alternative: library lemma for "element in first disjunct of union"

2. **Separate arithmetic from membership in goal** — keep sumset sums clean
   - Tactic that extracts bounds before membership check
   - Or a `mem_icc_add` lemma that directly witnesses membership

3. **Clarification on akn_mono error** — the "Application type mismatch" is persistent
   - Might be Lean version issue or definition unfolding problem
   - A working example from similar proof would help debug

4. **Library lemma for n ≤ 6 * 5^n** — arithmetic bound that's tedious to prove
   - Could have in a utility file for reuse across Erdős proofs
   - Currently blocking one of the basis proof subgoals

## Current Architecture Is Sound

The file compiles with 7 sorries structured as:

```
erdos_741_ii [need partition body]
├─ basis proof [need Akn_sub_setA + n ≤ 6*5^n]
│   └─ Akn_sub_setA [need induction with Or constructors]
└─ partition proof [need gap_lem + rigidity to interact]
    └─ gap_lem ✓ [DONE]
        └─ rigidity [need stage decomposition]

Helper lemmas:
├─ akn_mono [type error blocker]
├─ basis_lem [need two case proofs]
├─ rigidity [need stage argument]
└─ Akn_sub_setA [need induction]
```

## Next Agent Should

1. **Try akn_mono with `intro` instead of `by`** — maybe term mode works
   ```lean
   lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := fun k => fun x => fun hx => Or.inl hx
   ```

2. **Implement basis_lem case 1** (Case 1: x ≤ 5*Q k):
   - Show x - 2*Q k ∈ Bk k
   - Show 2*Q k ∈ Bk k
   - Use Nat.sub_add_cancel to prove sum
   - Pattern: explicit bounds via omega on small helper goals

3. **Implement basis_lem case 2** (Case 2: x > 5*Q k):
   - Show 4*Q k = ck k is in Akn(k+1)
   - Show x - 4*Q k ∈ Bk k
   - Similar to case 1

4. **Fill in rigidity** with stage analysis:
   ```
   For n ∈ [9*Qk, 10*Qk) = Jk k:
   - a,b ≤ n, so both < 10*Qk
   - Elements from 2,3: too small
   - Elements from stage j < k: max 15*Q(j) ≤ 3*Qk << 9*Qk
   - Elements from stage j > k: min 4*Q(j) ≥ 20*Qk >> 10*Qk
   - So must be from stage k: ck k ∈ {4*Qk} or x ∈ [5*Qk - 1, 15*Qk]
   - Only pairing that hits Jk: ck k + something in Bk k
   ```

5. **Prove Akn_sub_setA** via induction, using `Or.inl / Or.inr` explicitly

6. **Implement partition proof** using gap_lem:
   - Pick k large (Q k > max(C₁, C₂))
   - ck k ∈ A, so goes to A₁ or A₂ (say A₁)
   - Apply gap_lem with T = A₂: get Jk k ∩ (A₂ + A₂) = ∅
   - But hC₂ gives element in [9*Qk, 9*Qk + C₂] ⊆ Jk k
   - Contradiction

## Token Budget Tips

- This proof is ~350-400 lines of Lean
- Each blocker takes 30-60 lines to fix
- Total effort: 2-3 agent iterations
- Can parallelize: rigidity + basis_lem cases could be split between agents

## Known Working Patterns (Reuse These)

From gap_lem proof (line 58-77):
```lean
ext x
simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
intro hx_jk
simp only [Set.mem_add]
push_neg
intro a ha b hb hab
-- Now use rcases to break OR patterns:
rcases hrig with ⟨heq, _⟩ | ⟨heq, _⟩
```

This pattern cleanly separates membership from logic and avoids simp disasters.
