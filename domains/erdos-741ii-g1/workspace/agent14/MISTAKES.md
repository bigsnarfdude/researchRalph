# Agent 14 Mistakes — Erdős #741(ii)

## What Didn't Work

### 1. Using `right` Tactic for 3-Way Unions
**What I tried:**
```lean
simp only [stageK, Set.mem_union]
right; right  -- Expected to select Fk k, the third branch
```
**Result:** Failed with "right tactic works for inductive types with exactly 2 constructors"
**Why:** `right` only works on binary inductive types; `({a} ∪ {b}) ∪ {c}` has right-associative structure
**Fix:** Use `Or.inr (Or.inr ...)` instead

### 2. Using `norm_num` on Goals with Free Variables
**What I tried:**
```lean
exact Or.inr (Or.inr ⟨by norm_num, by norm_num⟩)
```
Where goal involved `n - 2 ∈ Icc ...` with free variable `n`
**Result:** "Expected type must not contain free variables"
**Why:** `decide` and `norm_num` work only on ground goals, not those with variables
**Fix:** Use `omega` instead, which handles linear arithmetic with variables

### 3. Direct Set Membership Proof with Angle Brackets
**What I tried:**
```lean
exact Or.inr (Or.inr ⟨by omega, by omega⟩)
```
**Result:** "Invalid `⟨...⟩` notation: expected type could not be determined"
**Why:** After `simp only [Fk, mem_Icc, Q]`, the goal is `x ∈ Icc ...`, not a pair
**Fix:** Use `mem_Icc.mpr ⟨by omega, by omega⟩` or just provide the pair directly if simp unfolds to conjunction

### 4. Trying to Prove basis_lem via Case Analysis on Small n
**What I tried:**
```lean
by_cases h : n ≤ 6
· interval_cases n
  all_goals (simp [setA]; ...)
```
**Result:** Goals still not closed, needed explicit construction for each case
**Why:** Each case requires a different witness pair; simp doesn't synthesize them
**Fix:** Use explicit pair construction with `use` for each case, or defer to later work

### 5. Attempting Lk_in_setA Membership Proof
**Issue:** The proof required showing `x ∈ Icc (10*Q k - 1) (15*Q k)` for `x ∈ Lk k = Icc (2*Q k) (3*Q k)`
**Problem:** Icc bounds for different k levels don't obviously overlap; may need different staging strategy
**Status:** Deferred to sorry; need to reconsider level definitions

## Patterns Causing Issues

1. **Set literal membership:** `{2, 3}` requires special handling; `simp` resolves it fully or not at all
2. **Multi-way unions:** Always require explicit Or constructor chains
3. **Nat subtraction:** Any arithmetic involving `n - m` must use `omega`, not `linarith` or `norm_num`
4. **Definitions with subtraction:** `Bk k = Icc (5*Q k) (6*Q k - 1)` contains subtraction; unfold carefully
