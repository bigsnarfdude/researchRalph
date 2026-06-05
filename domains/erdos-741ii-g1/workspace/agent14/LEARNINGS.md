# Agent 14 Learnings — Erdős #741(ii)

## Status
- **File compiles:** YES (5 sorry statements remaining)
- **Score:** 0.0 (SCORE = 1.0 requires all sorry eliminated)
- **Progress:** Core proof structure complete; 5 key lemmas need proof

## Key Findings

### 1. Lean 4 Union Type Handling
- `Set.mem_union` and membership in unions with 3+ branches requires careful handling
- `right` tactic only works for binary inductive types (2 constructors)
- For 3-way unions like `a ∪ b ∪ c`, must use `Or.inr (Or.inr ...)` explicitly
- Pattern: `{ck k} ∪ Bk k ∪ Fk k` parses as `({ck k} ∪ Bk k) ∪ Fk k`

### 2. Nat Subtraction in Lean
- As noted in mathlib_hints, `omega` is required for `ℕ` subtraction, not `linarith`
- This applies to Icc/Ico intervals like `Icc (6*Qk - 1) (15*Qk)`
- `omega` can close arithmetic goals involving these subtraction expressions

### 3. Membership Proof Construction
- Using `mem_Icc.mpr` works better than angle bracket notation with simp
- Pattern for proving `x ∈ Icc a b`: provide `⟨h_low, h_high⟩` to `mem_Icc.mpr`
- `simp` can fully resolve trivial set membership goals (e.g., `2 ∈ {2,3}`)
- Multiple `simp` calls in sequence can lead to "No goals to be solved" errors

### 4. Induction on `j ≤ k`
- Direct `induction` on `h : j ≤ k` is tricky—the bound variable is shadowed in step case
- Better: use `induction k` with case split, or use `generalizing j`
- Pattern that works: `induction k generalizing j` with by_cases on j

### 5. Proof Complexity
- Basis lemma (all n ≥ 4 in sum form) requires case-by-case analysis
- Q(0) = 1 means stage 0 covers only small intervals; need multiple stages for all n
- Gap lemma + rigidity are interdependent; both needed for main theorem

## Lemmas Proven (no sorry)
1. `Q_pos`: Q is strictly positive
2. `Q_succ`: Q(k+1) = 5 * Q(k)
3. `Q_mono`: Q is strictly increasing
4. `akn_mono`: Akn is monotone in its index

## Lemmas with sorry (5 remaining)
1. **basis_lem** (line 63): ∀ n ≥ 4, ∃ a, b ∈ A, a + b = n
2. **Lk_in_setA** (line 66): Level k fillers are in setA
3. **rigidity** (line 73): For n ∈ Jk, only specific pair types sum to n
4. **gap_lem** (line 77): If ck k ∉ T, then Jk ∩ (T + T) = ∅
5. **erdos_741_ii** (line 91): Main theorem (no both-syndetic partition exists)
