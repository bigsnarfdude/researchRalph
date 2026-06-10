import Mathlib
import Mathlib.Tactic.Linarith

def IsSolution (f : ℕ → ℕ) : Prop :=
  f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1))

def b (f : ℕ → ℕ) (n : ℕ) : ℤ := (f (n + 1) : ℤ) - (n + 1) - 1

theorem f_gt_n (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : n > 0 → f n > n := by
  intro hn
  induction n, hn using Nat.le_induction with
  | base => rw [h.1]; omega
  | succ m hm ih =>
    have : f (m + 1) > f m := h.2.2 m
    omega

theorem f0_eq_0 (f : ℕ → ℕ) (h : IsSolution f) : f 0 = 0 := by
  have hff0 := h.2.1 0
  by_contra h0
  have h0_pos : f 0 > 0 := Nat.pos_of_ne_zero h0
  have h_gt := f_gt_n f h (f 0) h0_pos
  rw [hff0] at h_gt
  omega

theorem f_pos (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : n > 0 → f n > 0 := by
  intro hn; have := f_gt_n f h n hn; omega

theorem b_jump (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  n > 0 → b f (f n - 1) = n - 1 := by
  intro hn
  unfold b
  have hfn : f n > 0 := f_pos f h n hn
  have h_eq : f n - 1 + 1 = f n := Nat.sub_add_cancel hfn
  rw [h_eq, h.2.1 n]
  omega

theorem b_mono (f : ℕ → ℕ) (h : IsSolution f) (m k : ℕ) (hmk : m ≤ k) :
  b f m ≤ b f k := by
  induction k, hmk using Nat.le_induction with
  | base => exact le_refl _
  | succ k' hk' ih =>
    apply ih.trans
    unfold b
    have h_inc := h.2.2 (k' + 1)
    omega

theorem f2_eq_3 (f : ℕ → ℕ) (h : IsSolution f) : f 2 = 3 := by
  have hf1 := h.1
  have hff1 := h.2.1 1
  rw [hf1] at hff1
  omega

-- Helper for monotonicity
theorem f_mono (f : ℕ → ℕ) (h : IsSolution f) (m n : ℕ) (hmn : m ≤ n) : f m ≤ f n := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_refl _
  | succ n' hn' ih =>
    apply ih.trans
    exact (h.2.2 n').le

theorem b_bounds (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  ∃ i, f i ≤ n ∧ n < f (i + 1) ∧ (b f n = (i : ℤ) - 1 ∨ b f n = (i : ℤ)) := by
  have h_exists : ∃ i, f i ≤ n ∧ n < f (i + 1) := by
    induction n with
    | zero => 
      use 0; rw [f0_eq_0 f h]
      constructor; exact Nat.le_refl _; exact f_pos f h 1 (by omega)
    | succ m ih_m =>
      obtain ⟨i, hi1, hi2⟩ := ih_m
      if h_lt : m + 1 < f (i + 1) then
        use i; constructor; exact hi1.trans (Nat.le_succ m); exact h_lt
      else
        have h_eq : m + 1 = f (i + 1) := by omega
        use i + 1; constructor
        · rw [h_eq]
        · rw [h_eq]; exact h.2.2 (i + 1)
  
  obtain ⟨i, hi1, hi2⟩ := h_exists
  use i; constructor; exact hi1; constructor; exact hi2
  
  if hi0 : i = 0 then
    subst hi0
    have hf1 : f 1 = 2 := h.1
    have hf2 : f 2 = 3 := f2_eq_3 f h
    have hb0 : b f 0 = 0 := by unfold b; rw [hf1]; omega
    have hb1 : b f 1 = 0 := by unfold b; rw [hf2]; omega
    have h_low : 0 ≤ b f n := by
       have := b_mono f h 0 n (Nat.zero_le n)
       rwa [hb0] at this
    have h_high : b f n ≤ 0 := by
       have h_le : n ≤ 1 := by rw [hf1] at hi2; omega
       have := b_mono f h n 1 h_le
       rwa [hb1] at this
    omega
  else
    have hi_pos : i > 0 := Nat.pos_of_ne_zero hi0
    have h_fi_pos : f i > 0 := f_pos f h i hi_pos
    have h_low : b f (f i - 1) ≤ b f n := b_mono f h (f i - 1) n (by omega)
    have h_high : b f n ≤ b f (f (i + 1) - 1) := b_mono f h n (f (i + 1) - 1) (by omega)
    have hb_low : b f (f i - 1) = (i : ℤ) - 1 := b_jump f h i hi_pos
    have hb_high : b f (f (i + 1) - 1) = (i : ℤ) := by
       have := b_jump f h (i + 1) (by omega)
       simp at this
       exact this
    rw [hb_low] at h_low
    rw [hb_high] at h_high
    omega

theorem count_diff_bound (f g : ℕ → ℕ) (hf : IsSolution f) (hg : IsSolution g) (n : ℕ)
  (ih : ∀ k, k ≤ n + 2 → |(f k : ℤ) - (g k : ℤ)| ≤ 1)
  (i j : ℕ) (hi1 : f i ≤ n) (hi2 : n < f (i + 1)) (hj1 : g j ≤ n) (hj2 : n < g (j + 1)) :
  | (i : ℤ) - (j : ℤ) | ≤ 1 := by
  by_contra h_contr
  simp at h_contr
  if h_lt : (i : ℤ) < (j : ℤ) - 1 then
    have h_j_ge : j ≥ i + 2 := by omega
    have h_g_i2 : g (i + 2) ≤ g j := f_mono g hg (i + 2) j h_j_ge
    have h_g_le : g (i + 2) ≤ n := h_g_i2.trans hj1
    have h_f_lt : n < f (i + 1) := hi2
    have h_le_bad : g (i + 2) < f (i + 1) := h_g_le.trans_lt h_f_lt
    
    have h_diff : | (f (i + 2) : ℤ) - (g (i + 2) : ℤ) | ≤ 1 := ih (i + 2) (by
       rcases Nat.eq_zero_or_pos i with rfl | hi_pos
       · omega
       · have : i + 1 ≤ f i := f_gt_n f hf i hi_pos
         omega)
    
    have h_f_ge : (f (i + 2) : ℤ) ≥ (g (i + 2) : ℤ) - 1 := by omega
    have h_f_inc : f (i + 1) < f (i + 2) := hf.2.2 (i + 1)
    omega
  else
    have h_lt' : (j : ℤ) < (i : ℤ) - 1 := by omega
    have h_i_ge : i ≥ j + 2 := by omega
    have h_f_j2 : f (j + 2) ≤ f i := f_mono f hf (j + 2) i h_i_ge
    have h_f_le : f (j + 2) ≤ n := h_f_j2.trans hi1
    have h_g_lt : n < g (j + 1) := hj2
    have h_le_bad : f (j + 2) < g (j + 1) := h_f_le.trans_lt h_g_lt
    
    have h_diff : | (f (j + 2) : ℤ) - (g (j + 2) : ℤ) | ≤ 1 := ih (j + 2) (by
       rcases Nat.eq_zero_or_pos j with rfl | hj_pos
       · omega
       · have : j + 1 ≤ g j := f_gt_n g hg j hj_pos
         omega)
    
    have h_g_ge : (g (j + 2) : ℤ) ≥ (f (j + 2) : ℤ) - 1 := by omega
    have h_g_inc : g (j + 1) < g (j + 2) := hg.2.2 (j + 1)
    omega

theorem stability_full (f g : ℕ → ℕ) (hf : IsSolution f) (hg : IsSolution g) (n : ℕ) :
  | (f n : ℤ) - (g n : ℤ) | ≤ 1 := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    if hn0 : n = 0 then rw [f0_eq_0 f hf, f0_eq_0 g hg]; simp
    else if hn1 : n = 1 then rw [hf.1, hg.1]; simp
    else
      have h_n_pos : n > 0 := Nat.pos_of_ne_zero hn0
      obtain ⟨i, hfi, hfi1, hb_f⟩ := b_bounds f hf (n-1)
      obtain ⟨j, hgj, hgj1, hb_g⟩ := b_bounds g hg (n-1)
      
      have h_fn : (f n : ℤ) = (n - 1 : ℤ) + 2 + b f (n - 1) := by unfold b; simp; omega
      have h_gn : (g n : ℤ) = (n - 1 : ℤ) + 2 + b g (n - 1) := by unfold b; simp; omega
      
      have h_diff : | (f n : ℤ) - (g n : ℤ) | = | b f (n - 1) - b g (n - 1) | := by
        rw [h_fn, h_gn]; ring_nf; simp
      
      have h_ij : | (i : ℤ) - (j : ℤ) | ≤ 1 := by
        apply count_diff_bound f g hf hg (n-1) _ i j hfi hfi1 hgj hgj1
        intro k hk
        -- We need k \le (n-1)+2 = n+1.
        -- But ih only gives us k < n.
        -- This is the tricky part: we need to assume IH up to n, and prove for n.
        -- But count stability at n-1 depends on stability at i+2, which can be > n-1.
        sorry
      sorry
