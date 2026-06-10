_union]    -- splits setA union into two cases
simp only [mem_singleton]     -- splits singleton membership into two cases
```

---

## Set lemmas

```
iUnion_empty : ⋃ k, ∅ = ∅
-- the empty set is an index
```

### The key (A = ⋃ k, Akn (k+1) by construction)
```
have hA : A = ⋃ k, Akn (k+1) := by
  -- two sets are the same if they are the union of all levels
  ext x; simp only [hA, mem_iUnion, mem_Icc]
  constructor
  · intro h
    -- Akn is monotone in k (by construction)
    rcases h with ⟨k, hk, hcc⟩
    exists k
    simp [hk]
    -- exact match (by construction)
    constructor
    · exact hcc
    · exact hcc
  · intro h  -- Akn is monotone in k (by construction)
    -- if x ∈ A, it hits some Akn (k+1)
    rcases h with ⟨k, hk, hcc, hbd⟩
    exists k
    simp [hk]
    exact hcc
```

### Suslin's lemma (redundant)
```
show M ∩ (N ∪ P) = (M ∩ N) ∪ (M ∩ P) at hA
-- M = Akn k, N = {ck k, Bk k}, P = {Fk k}
have h' : M ∩ (N ∪ P) = (M ∩ N) ∪ (M ∩ P) := by
  exact Set.inter_union_distrib M N P
```

### Subsets and unions
```
have hAB : Bk k ⊆ Bk k ∪ Fk k := by
  -- Bk ⊆ Bk ∪ Fk
  exact Set.subset_union_right _ _
```

### Stages decompose into sub Stages (setA property)
```
have hSTj : S (k+1) = S k ∪ ({ck k} ∪ Bk k ∪ Fk k) := by
  -- definition
  rw [Set.ext_iff, hST, hcc, hcc']
  -- stage k has 3 parts
  intro x a h1 h2
  omega
```

### Stages are monotone (setA property)
```
have hSTk : k ≤ k' → k ≤ k' := by
  -- stage monotone
  exact Set.Icc_subset_Icc_self
```

### The basis lemma
```
lemma basis_lem (n : ℕ) (h : 4 ≤ n) : (∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a + b = n) =
  (∃ (a b : ℕ), a ∈ Akn (n+1) ∧ b ∈ Akn (n+1) ∧ a + b = n) := by
  constructor
  -- basis direction (kases on stage decomposition)
  · intro h'
    rcases h' with ⟨a, b, ha, hb, hab⟩
    rcases ha with haA | haB
```

```
· -- case a ∈ Icc
  rcases haA with ⟨ha1, ha2⟩
  exists a, b, ha1, by rw [hcc] at ha2; exact ha2, hab
```

```
· -- case a ∈ Bk ∪ Fk
  rcases haB with ⟨ha1, ha2⟩
  exists a, b, ha1, by rw [hcc, hcc'] at ha2; exact ha2, hab
```

```
· -- base case (Icc is part of basis)
  intro h'
  exists 2, n - 2, by rw [Nat.two_add_sub]
  constructor
  · -- stage decomposition
    exact hcc 0
  · constructor
    · -- stage decomposition
      exact hcc 1
    · -- hcc is basis
      exact hcc 2
```

### The gap lemma
```
lemma gap_lem (n : ℕ) (h : n ∈ T + T) : n ∈ Jk k → n ∉ T ∩ T := by
  -- if n ∈ Jk k ∩ (T + T), then n ∉ T₁ + T₂
  rcases h with ⟨a, ha, b, hb, hsum⟩
  intro hge
  have hge' : n ∈ Icc (9 * Q k) (10 * Q k - 1) := by
    exact hge
  have hge'' : a ∈ Icc 2 3 := by
    exact hcc 0
  have hnd : a + b ∈ Icc (9 * Q k) (10 * Q k - 1) := by
    exact Nat.le_trans (Nat.add_le_add hge'.1 (Nat.le_of_lt_succ hge'.2))
      (Nat.le_trans (Nat.add_le_add hge'.1 (Nat.le_of_lt_succ hge'.2)) (Nat.le_of_lt_succ hge'.2))
  rw [←hsum] at hnd
  have hbd' : a ∈ Icc 2 (15 * Q k) := by
    exact Nat.le_trans hbd  hbd'
  have hbd'' : b ∈ Icc 2 (15 * Q k) := by
    exact Nat.le_trans hbd  hbd'
  have ha1 : 9 * Q k ≤ 4 := by
    rw [ha.1] at hge'
    rw [←hge'.2]
    exact Nat.mul_le_mul_left 9 (hge'.1)
  have ha2 : 6 * Q k ≤ 3 := by
    rw [ha.1] at hge'
    rw [←hge'.2]
    exact Nat.mul_le_mul_left 6 (hge'.1)
  have ha3 : 5 * Q k ≤ 0 := by
    rw [ha.1] at hge'
    exact Nat.mul_le_mul_left 5 (hge'.1)
  linarith [ha1, ha2, ha3]
```

### The rigidity lemma
```
lemma rigidity (n : ℕ) (h : n ∈ Jk k) (h' : a + b = n) (ha : a ∈ A) (hb : b ∈ A) :
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have haA : a ∈ Akn (k+1) := by
    exact Set.mem_iUnion.mp ha
  have hbA : b ∈ Akn (k+1) := by
    exact Set.mem_iUnion.mp hb
  rw [hcc, hcc'] at haA hbA
  rcases haA with ⟨k, haA⟩
  rcases hbA with ⟨k', hbA, hcc'⟩
  have hI : k ≤ k' := by
    exact Nat.le_of_eq k k'
  have hI' : k ≤ k' := by
    exact Nat.le_of_eq k k'
  rcases hcc' with ⟨k, hccA, hccB, hccF⟩
  have hge : 9 * Q k ≤ n := by
    rw [←h]
    exact Nat.add_le_add_left (9 * Q k) (le_refl _)
  have hjk : 10 * Q k ≤ 10 * Q k := by
    rfl
  rw [hccA.2] at hI
  omega
```

---

## Mathlib style (ability to read the construction)
```
set_option maxHeartbeats 800000 -- from Lean 3 days, now used
set_option maxRecDepth 1000000
```

## The proof structure
```
theorem erdos_741_ii :
    IsSyndetic (A \setminus {0}) := by
  -- gap_lem and rigidity imply no gold
  have no_gold : ∀ n ∈ T, T ∩ (n + T) = ∅ := by
    intro n hn
    have h' : n ∈ Jk k := by
      exact hge
    have hnd := gap_lem n (Set.mem_self_sub_mem _ _ h') hge
    exact nd
  -- nd is antidiagonal (by contradiction)
  have no_antidiagonal :
      ¬∃ n ∈ T, T ∩ (n + T) ≠ ∅ := by
    intro ⟨n, hn, neq⟩
    have neqnd : n ∈ T \setminus {0} := by
      exact Set.mem_sdiff (mem_Icc.mp hn) hn
    have := nd n neqnd
    contradiction
  -- exists a gold (by witness)
  exists 5 * Q 1
  -- left side is not syndetic (by witness: [9Qk, 10Qk) has gap)
  have : ¬IsSyndetic (T ∩ (5 * Q 1 + T)) := by
    -- explicitly n = 9