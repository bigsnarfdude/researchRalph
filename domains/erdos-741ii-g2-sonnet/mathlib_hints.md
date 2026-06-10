# Mathlib hints — Erdős #741(ii)

Extracted from the working proof. Every non-trivial API call, exact name, exact argument order.

---

## The single most important rule

**Use `omega` for any goal or hypothesis involving ℕ subtraction.**
`linarith` does NOT handle `n - m : ℕ`. It silently fails.
`Bk k = Icc (5*Qk) (6*Qk - 1)` and `Fk k = Icc (10*Qk-1) (15*Qk)` — both have nat-sub.
Whenever these appear in context, use `omega`.

---

## Arithmetic on ℕ

```
omega          -- closes linear arithmetic goals over ℕ, including nat-sub
linarith       -- closes linear arithmetic over ℤ/ℚ, NO nat-sub
norm_num       -- closes numeric ground goals: 0 < 5, 1 ≤ 5, etc.
pow_pos        -- pow_pos : 0 < a → 0 < a^n        (used for Q_pos)
```

### Nat.sub helpers (for sum-pair witnesses in basis_lem)
```
Nat.sub_add_cancel  : n ≤ m → m - n + n = m      -- (x-a) + a = x
Nat.add_sub_cancel' : n ≤ m → n + (m - n) = m    -- a + (x-a) = x  (flipped)
```
Pattern for exhibiting a pair that sums to x: `⟨x - a, hx_a_mem, a, ha_mem, Nat.sub_add_cancel hle⟩`

### Powers
```
Nat.pow_le_pow_right : 1 ≤ b → m ≤ n → b^m ≤ b^n
-- Used as: Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hj
-- to get: Q (j+1) ≤ Q k from j < k (i.e., j+1 ≤ k)
```

---

## Set membership

```
mem_Icc.mpr  : ⟨h_lo, h_hi⟩ → x ∈ Icc a b
mem_Icc.mp   : x ∈ Icc a b → a ≤ x ∧ x ≤ b
mem_Ico.mpr  : ⟨h_lo, h_hi⟩ → x ∈ Ico a b
mem_Ico.mp   : x ∈ Ico a b → a ≤ x ∧ x < b
Set.mem_add  : x ∈ (S + T) ↔ ∃ a ∈ S, ∃ b ∈ T, a + b = x
Set.mem_iUnion.mpr : ⟨k, hk⟩ → x ∈ ⋃ k, S k
mem_inter_iff : x ∈ S ∩ T ↔ x ∈ S ∧ x ∈ T
mem_empty_iff_false : x ∈ ∅ ↔ False
```

### Opening membership goals
```
simp only [mem_Icc] at hx         -- unfolds Icc membership to ≤ bounds
simp only [mem_Ico] at hx         -- unfolds Ico membership to ≤ and <
simp only [setA, mem_union, mem_iUnion] at ha  -- opens setA into its cases
simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx  -- opens stage membership
```

### Closing membership contradictions
```
simp [hdisj] at hmem   -- closes: hmem : x ∈ A₁ ∩ A₂ when hdisj : A₁ ∩ A₂ = ∅
simp [hgap] at hmem    -- closes: hmem : x ∈ Jk k ∩ (T+T) when hgap : ... = ∅
-- NOTE: Set.not_mem_empty does NOT exist in this Mathlib version. Use simp.
```

### Set subset/add
```
Set.add_subset_add : S₁ ⊆ S₂ → T₁ ⊆ T₂ → S₁ + T₁ ⊆ S₂ + T₂
Subset.trans / .trans  : S ⊆ T → T ⊆ U → S ⊆ U
```

---

## Proof structure patterns

### Case split on index vs k
```
rcases lt_trichotomy j k with hlt | hje | hgt
-- hlt : j < k  →  use small_stage
-- hje : j = k  →  use rw [hje] at haj  (NOT subst, NOT rcases | rfl |)
-- hgt : k < j  →  use large_stage
```

**CRITICAL — j = k branch:** Use `rw [hje] at haj`, never `subst hje` or `rcases ... | rfl | ...`.
`subst` and `rfl` both replace the outer parameter `k` with the fresh local `j`, making all
subsequent explicit `k` references fail as "Unknown identifier k".
`rw [hje] at haj` rewrites only the one hypothesis, `k` stays in scope everywhere.

### Destructor patterns
```
obtain ⟨hlo, hhi⟩ := hn             -- split a conjunction / Ico membership
obtain ⟨a, ha, b, hb, hab⟩ := h    -- split a sumset membership ∃ a ∈ S, ∃ b ∈ T, ...
rcases hx with ((rfl | ⟨h, _⟩) | ⟨h, _⟩)  -- 3-way split of stage membership
rintro ⟨a, ha, b, hb, hab⟩         -- intro + immediate destruct
```

### Induction
```
induction k with
| zero   => ...
| succ k ih => ...

induction h with       -- induction on a proof of m ≤ n
| refl    => rfl
| @step k _ ih => ...
```

### by_cases for interval coverage (basis_lem)
```
by_cases h1 : x ≤ 5 * Q k
· exact ⟨x - 2*Q k, inI _ (by omega) (by omega), 2*Q k, inI _ (by omega) (by omega),
         Nat.sub_add_cancel (by linarith)⟩
-- pattern: ⟨left_witness, left_mem, right_witness, right_mem, sum_proof⟩
```

---

## Opening / unfolding

```
unfold Q at *          -- unfolds def Q (k) := 5^k
simp [Q, pow_succ, mul_comm]   -- proves Q (k+1) = 5 * Q k
show T                 -- change goal to a definitionally equal T
push_neg at h          -- turns ¬(a ≤ b) into b < a
```

---

## File header (copy exactly)

```lean
import Mathlib
set_option maxHeartbeats 800000
set_option maxRecDepth 1000
open Set
open scoped Pointwise Classical BigOperators
```
