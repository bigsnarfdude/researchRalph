# Agent3 Erdős 741(ii) Progress

## Accomplishments

### Construction Definitions (Complete ✅)
- `Q k = 5^k`: Exponential function
- `ck k = 4 * Q k`: Connector element
- `Bk k = [5*Qk, 6*Qk - 1]`: Body interval
- `Fk k = [10*Qk - 1, 15*Qk]`: Filler interval
- `Jk k = [9*Qk, 10*Qk)`: Gap zone (Ico not Icc)
- `stage_union k = {ck k} ∪ Bk k ∪ Fk k`: One construction level
- `setA = {2, 3} ∪ ⋃ k, stage_union k`: Full construction
- `Akn k = {2, 3} ∪ ⋃ j ≤ k, stage_union j`: Partial union through level k

### Helper Lemmas (Complete ✅)
- `Q_pos k`: Proves 0 < Q k using pow_pos
- `Q_succ k`: Proves Q(k+1) = 5 * Q k using pow_succ
- `akn_mono k`: Proves Akn k ⊆ Akn(k+1) using iUnion bounds

### IsSyndetic Definition (Complete ✅)
- Defined as: ∃ C, ∀ x, ∃ m ∈ S, m ∈ Icc x (x+C)
- Matches problem statement for density/hitting properties

### Main Theorem Structure (Complete ✅)
- Correctly structured existence proof with two goals:
  1. Basis property: ∀ n ≥ 4, ∃ a,b ∈ setA, a+b=n
  2. Rigidity/gap property: No partition is both-syndetic

## Remaining Sorries (5 remaining)

### 1. basis_lem (Lines 46-50)
**Statement**: `Icc 4 (6*Q k) ⊆ Akn(k+1) + Akn(k+1)`

**Strategy**: For x ∈ [4, 6*Q k], decompose as sum of two elements from Akn(k+1).
- Small x (4-6): Use pairs like 2+2=4, 2+3=5, 3+3=6
- Large x: Use structure of stage unions

**Challenge**: Need to handle membership proof for {2,3} set and show coverage for arbitrary interval.

### 2. rigidity_lem (Lines 53-69)
**Statement**: For n ∈ [9*Qk, 10*Qk) with a+b=n and a,b∈setA: (a=ck k ∧ b∈Bk k) ∨ (b=ck k ∧ a∈Bk k)

**Strategy**: Case analysis on which stages a and b come from.
- Elements from {2,3}: Too small (≤ 3)
- Elements from stage j << k: Bounded above by ~3*Q k
- Elements from stage j >> k: Bounded below by ~4*Q k
- Only stage k works: ck k + Bk k covers [9*Qk, 10*Qk)

**Challenge**: Requires many sub-cases for geometric bounds on different stage combinations.

### 3. gap_lem (Lines 72-74)
**Statement**: If ck k ∉ T ⊆ setA, then Jk k ∩ (T+T) = ∅

**Strategy**: By contradiction via rigidity:
1. Assume n ∈ Jk k ∩ (T+T)
2. Decompose n = a+b with a,b ∈ T
3. Apply rigidity to get one of {a,b} = ck k
4. Contradiction: ck k ∈ T contradicts hck_notin

**Challenge**: Proper handling of by_contra and converting between ¬(S = ∅) and (S ≠ ∅) and nonempty.

### 4 & 5. Main theorem parts (Lines 86-89, 90-93)
**First part**: Show all n ≥ 4 in setA + setA
- Depends on basis_lem via Akn ⊆ setA

**Second part**: No partition is both-syndetic  
- Key proof: Use gap_lem to create contradiction
- Pick k with Q k > max(C₁, C₂)
- Then ck k ∈ A must go to A₁ or A₂
- Say ck k ∈ A₁ → Jk k ∩ (A₂+A₂) = ∅
- But syndetic property puts elements there
- Contradiction

## Key Technical Insights

1. **Membership proofs for {2,3}**: The literal set syntax `{2, 3}` requires careful handling with `mem_singleton_iff` and union membership rules.

2. **Natural number subtraction**: All lemmas using Bk k and Fk k definitions contain (6*Qk - 1) and (10*Qk - 1), which require `omega` not `linarith` for nat-sub goals.

3. **Ico vs Icc**: Gap zone Jk k is Ico (half-open) not Icc. This is essential for the gap argument - the open upper bound prevents hitting Fk exactly.

4. **Stage isolation**: The geometric growth of Q ensures stage j ≠ k elements stay far from [9*Qk, 10*Qk).

## Next Steps

To complete SCORE=1.0:
1. Implement basis_lem: Structured case split on small/large x with explicit pair construction
2. Implement rigidity_lem: Exhaustive case analysis on stage combinations with omega/linarith
3. Implement gap_lem: Fix by_contra pattern to properly yield contradiction
4. Complete main theorem parts using gap_lem for final contradiction

All core proof strategies are sound; remaining work is tactical Lean machinery.
