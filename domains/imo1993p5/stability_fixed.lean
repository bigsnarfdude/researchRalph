import Mathlib
import Mathlib.Tactic.Linarith

def IsSolution (f : ℕ → ℕ) : Prop :=
  f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1))

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
  intro hn; have : f n > n := f_gt_n f h n hn; omega

theorem f_inc (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : f n < f (n + 1) := h.2.2 n

theorem f_jumps (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : 
  f (n + 1) - f n = 1 ∨ f (n + 1) - f n = 2 := sorry 

def b (f : ℕ → ℕ) (n : ℕ) : ℤ := (f (n + 1) : ℤ) - (n + 1) - 1

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
  have hf1 := h.1; have hff1 := h.2.1 1; rw [hf1] at hff1; omega

theorem f3_eq_5 (f : ℕ → ℕ) (h : IsSolution f) : f 3 = 5 := by
  have hf2 := f2_eq_3 f h; have hff2 := h.2.1 2; rw [hf2] at hff2; omega

theorem f5_eq_8 (f : ℕ → ℕ) (h : IsSolution f) : f 5 = 8 := by
  have hf3 := f3_eq_5 f h; have hff3 := h.2.1 3; rw [hf3] at hff3; omega

theorem f4_in_67 (f : ℕ → ℕ) (h : IsSolution f) : f 4 = 6 ∨ f 4 = 7 := by
  have f3 := f3_eq_5 f h; have f5 := f5_eq_8 f h
  have lt34 := h.2.2 3; have lt45 := h.2.2 4
  rw [f3] at lt34; rw [f5] at lt45
  interval_cases f 4 <;> simp

theorem f_mono (f : ℕ → ℕ) (h : IsSolution f) (m n : ℕ) (hmn : m ≤ n) : f m ≤ f n := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_refl _
  | succ n' hn' ih => apply ih.trans; exact (h.2.2 n').le

theorem b_bounds (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  ∃ i, f i ≤ n ∧ n < f (i + 1) ∧ (b f n = (i : ℤ) - 1 ∨ b f n = (i : ℤ)) := by
  have h_exists : ∃ i, f i ≤ n ∧ n < f (i + 1) := by
    induction n with
    | zero => use 0; rw [f0_eq_0 f h]; constructor; exact Nat.le_refl _; exact f_pos f h 1 (by omega)
    | succ m ih_m =>
      obtain ⟨i, hi1, hi2⟩ := ih_m
      if h_lt : m + 1 < f (i + 1) then use i; constructor; exact hi1.trans (Nat.le_succ m); exact h_lt
      else
        have h_eq : m + 1 = f (i + 1) := by omega
        use i + 1; constructor; rw [h_eq]; rw [h_eq]; exact h.2.2 (i + 1)
  obtain ⟨i, hi1, hi2⟩ := h_exists
  use i; constructor; exact hi1; constructor; exact hi2
  if hi0 : i = 0 then
    subst hi0; have hf1 : f 1 = 2 := h.1; have hf2 : f 2 = 3 := f2_eq_3 f h
    have hb0 : b f 0 = 0 := by unfold b; rw [hf1]; omega
    have hb1 : b f 1 = 0 := by unfold b; rw [hf2]; omega
    have h_le : n ≤ 1 := by rw [hf1] at hi2; omega
    have h_low := b_mono f h 0 n (Nat.zero_le n)
    have h_high := b_mono f h n 1 h_le
    rw [hb0] at h_low; rw [hb1] at h_high; linarith
  else
    have hi_pos : i > 0 := Nat.pos_of_ne_zero hi0
    have hb_low : b f (f i - 1) = (i : ℤ) - 1 := b_jump f h i hi_pos
    have hb_high : b f (f (i + 1) - 1) = (i : ℤ) := by
       have := b_jump f h (i + 1) (by omega); simp at this; exact this
    have h_low := b_mono f h (f i - 1) n (by omega)
    have h_high := b_mono f h n (f (i + 1) - 1) (by omega)
    rw [hb_low] at h_low; rw [hb_high] at h_high; linarith

theorem stability (f g : ℕ → ℕ) (hf : IsSolution f) (hg : IsSolution g) (n : ℕ) :
  |(f n : ℤ) - (g n : ℤ)| ≤ 1 := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    if hn0 : n = 0 then subst hn0; rw [f0_eq_0 f hf, f0_eq_0 g hg]; simp
    else if hn1 : n = 1 then subst hn1; rw [hf.1, hg.1]; simp
    else if hn2 : n = 2 then subst hn2; rw [f2_eq_3 f hf, f2_eq_3 g hg]; simp
    else if hn3 : n = 3 then subst hn3; rw [f3_eq_5 f hf, f3_eq_5 g hg]; simp
    else if hn4 : n = 4 then
      subst hn4; have hf4_v := f4_in_67 f hf; have hg4_v := f4_in_67 g hg
      rcases hf4_v with hf4 | hf4 <;> rcases hg4_v with hg4 | hg4 <;> (rw [hf4, hg4]; simp)
    else
      -- n >= 5
      obtain ⟨i, hfi, hfi1, hb_f⟩ := b_bounds f hf (n-1)
      obtain ⟨j, hgj, hgj1, hb_g⟩ := b_bounds g hg (n-1)
      
      have h_f_ge_m2 : ∀ m, m ≥ 3 → f m ≥ m + 2 := by
        intro m hm; induction m, hm using Nat.le_induction with
        | base => have := f3_eq_5 f hf; omega
        | succ m' hm' ih_m => have := f_inc f hf m'; omega
      have h_g_ge_m2 : ∀ m, m ≥ 3 → g m ≥ m + 2 := by
        intro m hm; induction m, hm using Nat.le_induction with
        | base => have := f3_eq_5 g hg; omega
        | succ m' hm' ih_m => have := f_inc g hg m'; omega

      have hi_bound : i + 2 < n := by
        if hi3 : i < 3 then interval_cases i <;> omega
        else have : f i ≥ i + 2 := h_f_ge_m2 i (by omega); omega
      have hj_bound : j + 2 < n := by
        if hj3 : j < 3 then interval_cases j <;> omega
        else have : g j ≥ j + 2 := h_g_ge_m2 j (by omega); omega

      have h_ij : |(i : ℤ) - (j : ℤ)| ≤ 1 := by
        rw [abs_le]; constructor
        · by_contra h_contr; have : (i : ℤ) - (j : ℤ) ≥ 2 := by linarith
          have h_i_ge : i ≥ j + 2 := by linarith
          have h_f_le : f (j + 2) ≤ n - 1 := (f_mono f hf (j + 2) i h_i_ge).trans hfi
          have h_g_lt : n - 1 < g (j + 1) := hgj1
          have h_ih := ih (j + 2) hj_bound
          have : g (j + 1) < g (j + 2) := hg.2.2 (j + 1)
          rw [abs_le] at h_ih; linarith
        · by_contra h_contr; have : (j : ℤ) - (i : ℤ) ≥ 2 := by linarith
          have h_j_ge : j ≥ i + 2 := by linarith
          have h_g_le : g (i + 2) ≤ n - 1 := (f_mono g hg (i + 2) j h_j_ge).trans hgj
          have h_f_lt : n - 1 < f (i + 1) := hfi1
          have h_ih := ih (i + 2) hi_bound
          have : f (i + 1) < f (i + 2) := hf.2.2 (i + 1)
          rw [abs_le] at h_ih; linarith

      have h_fn : (f n : ℤ) = (n : ℤ) + 1 + b f (n - 1) := by 
        unfold b; rw [Nat.sub_add_cancel (by omega)]; linarith
      have h_gn : (g n : ℤ) = (n : ℤ) + 1 + b g (n - 1) := by 
        unfold b; rw [Nat.sub_add_cancel (by omega)]; linarith

      rw [h_fn, h_gn]; have : (f n : ℤ) - (g n : ℤ) = b f (n - 1) - b g (n - 1) := by linarith
      rw [abs_le]; constructor
      · -- f n - g n <= 1
        by_contra h_contr; have h_diff : b f (n - 1) - b g (n - 1) ≥ 2 := by linarith
        have hi_j1 : i = j + 1 := by linarith
        have hbf : b f (n - 1) = (i : ℤ) := by linarith
        have hbg : b g (n - 1) = (j : ℤ) - 1 := by linarith
        
        have h_n_ge_fi : n - 1 ≥ f i := by
          by_contra h_lt; have : n - 1 ≤ f i - 1 := by linarith
          have : b f (n - 1) ≤ b f (f i - 1) := b_mono f hf (n - 1) (f i - 1) this
          rw [hbf, b_jump f hf i (by omega)] at this; linarith
        
        have : b g (g j) = (g (g j + 1) : ℤ) - g j - 2 := by unfold b; linarith
        have hg_jump := f_jumps g hg (g j)
        have hggj := hg.2.1 j
        have : b g (g j) = (j : ℤ) ∨ b g (g j) = (j : ℤ) - 1 := by
          rw [hggj] at hg_jump; linarith
        
        have h_n_lt_gj : n - 1 < g j := by
          by_contra h_ge; have : b g (n - 1) ≥ b g (g j) := b_mono g hg (g j) (n - 1) (by linarith)
          rw [hbg] at this; linarith
        
        have h_ih_j := ih j (by omega)
        have : f j < f (j + 1) := hf.2.2 j
        rw [abs_le] at h_ih_j; linarith

      · -- g n - f n <= 1
        by_contra h_contr; have h_diff : b g (n - 1) - b f (n - 1) ≥ 2 := by linarith
        have hj_i1 : j = i + 1 := by linarith
        have hbg : b g (n - 1) = (j : ℤ) := by linarith
        have hbf : b f (n - 1) = (i : ℤ) - 1 := by linarith
        
        have h_n_ge_gj : n - 1 ≥ g j := by
          by_contra h_lt; have : n - 1 ≤ g j - 1 := by linarith
          have : b g (n - 1) ≤ b g (g j - 1) := b_mono g hg (n - 1) (g j - 1) this
          rw [hbg, b_jump g hg j (by omega)] at this; linarith
        
        have : b f (f i) = (f (f i + 1) : ℤ) - f i - 2 := by unfold b; linarith
        have hf_jump := f_jumps f hf (f i)
        have hffi := hf.2.1 i
        have : b f (f i) = (i : ℤ) ∨ b f (f i) = (i : ℤ) - 1 := by
          rw [hffi] at hf_jump; linarith

        have h_n_lt_fi : n - 1 < f i := by
          by_contra h_ge; have : b f (n - 1) ≥ b f (f i) := b_mono f hf (f i) (n - 1) (by linarith)
          rw [hbf] at this; linarith
          
        have h_ih_i := ih i (by omega)
        have : g i < g (i + 1) := hg.2.2 i
        rw [abs_le] at h_ih_i; linarith
