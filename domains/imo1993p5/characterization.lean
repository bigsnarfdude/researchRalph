import Mathlib

/--
Problem: f(1)=2, f(f n) = f n + n, f strictly increasing.
Theorem: f(n+1) - f(n) ∈ {1, 2} for all n.
-/
theorem f_jumps (f : ℕ → ℕ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ n, f (f n) = f n + n) 
  (h3 : ∀ n, f n < f (n + 1)) : 
  ∀ n, f (n + 1) - f n = 1 ∨ f (n + 1) - f n = 2 := by
  let d (n : ℕ) : ℤ := (f (n + 1) : ℤ) - (f n : ℤ)
  
  have d_pos : ∀ n, 1 ≤ d n := by
    intro n; unfold d; simp; exact Nat.succ_le_iff.mpr (h3 n)
  
  have f_f_diff : ∀ n, (f (f (n + 1)) : ℤ) - (f (f n) : ℤ) = d n + 1 := by
    intro n; unfold d; simp [h2]; ring
  
  have telescoping : ∀ n, (Finset.range (f (n + 1) - f n)).sum (fun i => d (f n + i)) = d n + 1 := by
    intro n
    let k := f (n + 1) - f n
    have hk : (k : ℤ) = d n := by
      unfold d; simp; exact Nat.cast_sub (h3 n |>.le)
    have h_sum := Finset.sum_range_sub (fun i => (f (f n + i) : ℤ)) k
    simp at h_sum
    rw [← h_sum, ← hk]
    apply Finset.sum_congr rfl
    intro i _
    unfold d; simp
    have : f n + (i + 1) = f n + i + 1 := by omega
    rw [this]

  have d_le_two : ∀ n, d n ≤ 2 := by
    -- Suppose there exists some n such that d n >= 3.
    -- Then its children d(f n + i) must be small.
    -- We can use induction on the value of d n, but it's not a simple induction.
    -- Instead, let's use the fact that if d n >= 3, then 
    -- Sum_{i=0}^{d n - 1} d(f n + i) = d n + 1
    -- 3*1 <= Sum <= d n + 1? No.
    -- Let M = sup d n. Then M <= (M+1)/M? No.
    -- If d n <= M for all n, then d n + 1 <= M * d n.
    -- 1 <= (M-1) d n.
    -- If M >= 3, then 1 <= 2 d n which is always true.
    
    -- Let's use the property that f(n) -> infinity.
    -- If d n was 3, then the average of its children is (3+1)/3 = 4/3.
    -- This means there are more 1s than 2s.
    -- But if there are many 1s, their children will be (1+1)/1 = 2.
    -- So 1s produce 2s, and 2s produce (2+1)/2 = 1.5 (one 1 and one 2).
    -- This suggests the ratio of 2s to 1s should be phi.
    -- d n = 1 -> children: (2)
    -- d n = 2 -> children: (1, 2) or (2, 1)
    -- In both cases, the children are <= 2.
    -- So if d n <= 2 for some n, then its children are <= 2.
    -- Since d 1 = f 2 - f 1.
    -- f 1 = 2. f(f 1) = f 1 + 1 => f 2 = 3.
    -- d 1 = 3 - 2 = 1.
    -- Since d 1 = 1 <= 2, its children are <= 2.
    -- The children of d 1 is d(f 1) = d 2.
    -- So d 2 <= 2.
    -- We need to show that this "covers" all n.
    intro n
    have h_le_two : ∀ m, (∃ k, m = f k) → d m ≤ 2 := by
      intro m ⟨k, hk⟩
      rw [hk]
      -- We'll use induction on k.
      induction k using Nat.strongRecOn with
      | ind k ih =>
        let dk := f (k + 1) - f k
        have h_sum := telescoping k
        -- Sum_{i=0}^{dk-1} d(f k + i) = d k + 1
        -- If i = 0, we have d(f k)
        -- We want to show d(f k) <= 2.
        -- d(f k) + Sum_{i=1}^{dk-1} d(f k + i) = d k + 1
        -- d(f k) + (d k - 1) * 1 <= d k + 1 => d(f k) <= 2.
        have h_split := Finset.sum_eq_add_sum_diff_single 0 (Finset.mem_range.mpr (by omega))
        rw [h_split] at h_sum
        have h_lower : (Finset.range dk).erase 0 |>.sum (fun i => d (f k + i)) ≥ (dk - 1) := by
          apply Finset.card_nsmul_le_sum
          intro i hi
          exact d_pos (f k + i)
        have : dk = d k := by unfold d; simp; exact Nat.cast_sub (h3 k |>.le)
        -- Wait, dk is Nat, d k is Int.
        have : (dk : ℤ) = d k := by unfold d; simp; exact Nat.cast_sub (h3 k |>.le)
        omega
    
    -- Now we need to show that for any n, d n <= 2.
    -- Every n is either a value of f or not.
    -- Wait, if n is not a value of f, it's still a child of some d k.
    -- Specifically, if d k = 2, then f k + 1 is not a value of f (since f k < f k + 1 < f(k+1)).
    -- And d(f k + 1) is one of the terms in the telescoping sum for d k.
    -- So we just need to show that ALL terms in the telescoping sum are <= 2.
    -- Let's use induction on n.
    induction n using Nat.strongRecOn with
    | ind n ih =>
      -- Find k such that f k ≤ n < f(k+1)
      -- This k exists because f(0)=0 and f(n) -> infinity.
      have f0_eq_0 : f 0 = 0 := by
        have h := h2 0; simp at h
        by_contra h_nz
        have : 0 < f 0 := Nat.pos_of_ne_zero h_nz
        have : f 0 < f (f 0) := by
          induction f 0, this using Nat.le_induction with
          | base => exact h3 0
          | succ y hy ih_f => exact ih_f.trans (h3 y)
        omega
      
      have h_exists : ∃ k, f k ≤ n ∧ n < f (k + 1) := by
        -- Standard property of increasing sequences starting at 0
        induction n with
        | zero => use 0; simp [f0_eq_0, h3]
        | succ m ih_m =>
          obtain ⟨k, hk1, hk2⟩ := ih_m
          if h_lt : m + 1 < f (k + 1) then
            use k; constructor; exact hk1.trans (Nat.le_succ m); exact h_lt
          else
            have h_eq : m + 1 = f (k + 1) := by omega
            use k + 1; constructor; rw [h_eq]; exact h3 (k + 1)
      
      obtain ⟨k, hk1, hk2⟩ := h_exists
      let dk := f (k + 1) - f k
      have h_sum := telescoping k
      -- Sum_{i=0}^{dk-1} d(f k + i) = d k + 1
      -- n = f k + i for some i < dk
      let i := n - f k
      have hi : i < dk := by omega
      have hn_eq : n = f k + i := by omega
      
      -- d n + Sum_{j ≠ i} d (f k + j) = d k + 1
      -- d n + (dk - 1) * 1 ≤ d k + 1
      -- d n + dk - 1 ≤ d k + 1
      -- Since dk = d k, d n ≤ 2.
      have h_split := Finset.sum_eq_add_sum_diff_single i (Finset.mem_range.mpr hi)
      rw [hn_eq, h_split] at h_sum
      have h_lower : (Finset.range dk).erase i |>.sum (fun j => d (f k + j)) ≥ (dk - 1) := by
        apply Finset.card_nsmul_le_sum
        intro j hj
        exact d_pos (f k + j)
      have : (dk : ℤ) = d k := by unfold d; simp; exact Nat.cast_sub (h3 k |>.le)
      omega

  intro n
  have h_le := d_le_two n
  have h_ge := d_pos n
  -- d n = 1 or d n = 2
  have : d n = 1 ∨ d n = 2 := by omega
  unfold d at this
  simp at this
  exact this
