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

theorem f_inc (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : f n < f (n + 1) := h.2.2 n

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

theorem f3_eq_5 (f : ℕ → ℕ) (h : IsSolution f) : f 3 = 5 := by
  have hf2 := f2_eq_3 f h
  have hff2 := h.2.1 2
  rw [hf2] at hff2
  omega

theorem f5_eq_8 (f : ℕ → ℕ) (h : IsSolution f) : f 5 = 8 := by
  have hf3 := f3_eq_5 f h
  have hff3 := h.2.1 3
  rw [hf3] at hff3
  omega

theorem f4_in_67 (f : ℕ → ℕ) (h : IsSolution f) : f 4 = 6 ∨ f 4 = 7 := by
  have f3 := f3_eq_5 f h
  have f5 := f5_eq_8 f h
  have lt34 := h.2.2 3
  have lt45 := h.2.2 4
  rw [f3] at lt34
  rw [f5] at lt45
  interval_cases f 4
  · left; rfl
  · right; rfl

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

theorem stability (f g : ℕ → ℕ) (hf : IsSolution f) (hg : IsSolution g) (n : ℕ) :
  (f n : ℤ) - (g n : ℤ) ≤ 1 ∧ (g n : ℤ) - (f n : ℤ) ≤ 1 := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    if hn0 : n = 0 then subst hn0; rw [f0_eq_0 f hf, f0_eq_0 g hg]; constructor <;> omega
    else if hn1 : n = 1 then subst hn1; rw [hf.1, hg.1]; constructor <;> omega
    else if hn2 : n = 2 then subst hn2; rw [f2_eq_3 f hf, f2_eq_3 g hg]; constructor <;> omega
    else if hn3 : n = 3 then subst hn3; rw [f3_eq_5 f hf, f3_eq_5 g hg]; constructor <;> omega
    else if hn4 : n = 4 then
      subst hn4
      have hf4_v := f4_in_67 f hf
      have hg4_v := f4_in_67 g hg
      rcases hf4_v with hf4 | hf4 <;> rcases hg4_v with hg4 | hg4 <;> (rw [hf4, hg4]; constructor <;> omega)
    else
      -- n >= 5
      obtain ⟨i, hfi, hfi1, hb_f⟩ := b_bounds f hf (n-1)
      obtain ⟨j, hgj, hgj1, hb_g⟩ := b_bounds g hg (n-1)
      
      have hi_bound : i + 2 < n := by
        have h_f3 : f 3 = 5 := f3_eq_5 f hf
        have h_fn2 : ∀ m, m ≥ 3 → f m ≥ m + 2 := by
          intro m hm
          induction m, hm using Nat.le_induction with
          | base => exact h_f3.ge
          | succ m' hm' ih_m =>
            have : f (m' + 1) ≥ f m' + 1 := by
              have := f_inc f hf m'; omega
            omega
        if hi3 : i < 3 then interval_cases i <;> omega
        else have : f i ≥ i + 2 := h_fn2 i (by omega); omega
      
      have hj_bound : j + 2 < n := by
        have h_g3 : g 3 = 5 := f3_eq_5 g hg
        have h_gn2 : ∀ m, m ≥ 3 → g m ≥ m + 2 := by
          intro m hm
          induction m, hm using Nat.le_induction with
          | base => exact h_g3.ge
          | succ m' hm' ih_m =>
            have : g (m' + 1) ≥ g m' + 1 := by
              have := f_inc g hg m'; omega
            omega
        if hj3 : j < 3 then interval_cases j <;> omega
        else have : g j ≥ j + 2 := h_gn2 j (by omega); omega

      have h_ij : (i : ℤ) - (j : ℤ) ≤ 1 ∧ (j : ℤ) - (i : ℤ) ≤ 1 := by
        constructor
        · by_contra h_contr
          have h_i_ge : i ≥ j + 2 := by omega
          have h_f_j2 : f (j + 2) ≤ f i := f_mono f hf (j + 2) i h_i_ge
          have h_f_le : f (j + 2) ≤ n - 1 := h_f_j2.trans hfi
          have h_g_lt : n - 1 < g (j + 1) := hgj1
          have h_diff : (g (j + 2) : ℤ) - (f (j + 2) : ℤ) ≤ 1 := (ih (j + 2) hj_bound).2
          omega
        · by_contra h_contr
          have h_j_ge : j ≥ i + 2 := by omega
          have h_g_i2 : g (i + 2) ≤ g j := f_mono g hg (i + 2) j h_j_ge
          have h_g_le : g (i + 2) ≤ n - 1 := h_g_i2.trans hgj
          have h_f_lt : n - 1 < f (i + 1) := hfi1
          have h_diff : (f (i + 2) : ℤ) - (g (i + 2) : ℤ) ≤ 1 := (ih (i + 2) hi_bound).1
          omega

      have h_fn : (f n : ℤ) = (n : ℤ) + 1 + b f (n - 1) := by 
        unfold b; have : n - 1 + 1 = n := Nat.sub_add_cancel (by omega); omega
      have h_gn : (g n : ℤ) = (n : ℤ) + 1 + b g (n - 1) := by 
        unfold b; have : n - 1 + 1 = n := Nat.sub_add_cancel (by omega); omega

      rw [h_fn, h_gn]
      rcases hb_f with hbf | hbf <;> rcases hb_g with hbg | hbg <;> omega
