# Agent11 Progress on Erdős #741(ii)

## Summary
Session focused on constructing the full proof scaffold for the set A = {2,3} ∪ ⋃_k ({ck k} ∪ Bk k ∪ Fk k) using Q k = 5^k.

## Completed Proofs

### 1. **Q_pos, Q_succ** (2/2)
- Q k = 5^k is always positive
- Q(k+1) = 5*Q(k)

### 2. **akn_mono** ✓
The partial union Akn k ⊆ Akn (k+1) is monotonic. Proved by induction:
- Base: Akn 0 = {2,3} ⊆ Akn 1 = {2,3} ∪ ... (by union property)
- Step: If x ∈ Akn k then x ∈ Akn(k+1) ⊆ Akn(k+2) (by ih + union)

### 3. **gap_lem** ✓
The gap zone Jk k = [9*Qk, 10*Qk) is disjoint from T+T when ck k ∉ T.
Proof: Any x ∈ Jk k ∩ (T+T) must factor as a+b ∈ T+T ∩ Jk k. By rigidity, this forces ck k ∈ {a,b}, contradicting hc.

### 4. **Q unbounded** ✓
Proved ∃k, max C₁ C₂ < Q k by using k = max C₁ C₂ + 1 and omega.

### 5. **ck k ∈ setA** ✓
Direct by unfolding setA definition and using right constructor + singleton.

## Remaining Sorries (4)

### 1. **basis_lem** (line 54)
Proves ∀n ∈ [4, 6*Qk], ∃a,b ∈ Akn(k+1), a+b=n.
- Requires 8-case analysis for different regions
- Cases: I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk
- Currently deferred; structure and tactics ready

### 2. **rigidity** (line 65)
Proves ∀a,b ∈ setA with a+b ∈ Jk k, then ck k ∈ {a,b}.
- Decompose elements by their source level (stage)
- Elements from {2,3}: too small (≤3 << 9*Qk)
- Stage j < k: max 15*Q(j) ≤ 3*Qk (geometric decay)
- Stage j > k: min 4*Q(j) ≥ 20*Qk > 10*Qk (impossible)
- Stage j = k: only ck k + Bk k sums into Jk k
- Currently deferred; argument structure complete

### 3. **Basis coverage** (line 93)
First part of main theorem: ∀n ≥ 4, ∃a,b ∈ setA, a+b=n.
- Find k with n ≤ 6*Qk (via Q's exponential growth)
- Use basis_lem to get decomposition in Akn(k+1)
- Use monotonicity + definition to get membership in setA
- Currently deferred; proof outline complete

### 4. **Symmetric case** (line 129)
Handles the case where ck k ∈ A₂. Argument mirrors the ck k ∈ A₁ case (swapping indices).
- Currently a placeholder sorry
- Can be filled with symmetric versions of the A₁ case

## Architecture Notes

**Main theorem structure:**
- Use setA as the witness for existence
- Split into two goals: (1) basis property, (2) no partition is both-syndetic
- Goal (2) uses gap_lem as the contradiction engine
- The key invariant: ck k must go to one partition, forcing its counterpart empty in sumsets

**Lean technical notes:**
- `gap_lem` uses `ext x` to convert set equality to pointwise negation
- `akn_mono` uses `simp only [Akn]` to unfold recursive definition
- Set membership destruction via `obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add ...`
- Interval subset proofs use `omega` for ℕ arithmetic (NOT linarith)

## Next Steps

1. **Prove basis_lem**: Implement the 8-case interval analysis
2. **Prove rigidity**: Case-split on element sources and bound arithmetic
3. **Fill basis coverage**: Chain lemmas with monotonicity
4. **Complete symmetric case**: Mirror the A₁ argument

**Estimated complexity:**
- basis_lem: ~30-40 lines (case-by-case)
- rigidity: ~25-30 lines (stage decomposition + bounds)
- Coverage + symmetric: ~10-15 lines (straightforward application)

Total path to SCORE=1.0: ~65-85 more lines of tactic proof.
