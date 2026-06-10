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
  intro hn; have := f_gt_n f h n hn; omega

theorem f_inc (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : f n < f (n + 1) := h.2.2 n

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

def b (f : ℕ → ℕ) (n : ℕ) : ℤ := (f (n + 1) : ℤ) - (n + 1) - 1

theorem b_jump (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  n > 0 → b f (f n - 1) = (n : ℤ) - 1 := by
  intro hn
  unfold b
  have hfn : f n > 0 := f_pos f h n hn
  have h_eq : f n - 1 + 1 = f n := Nat.sub_add_cancel hfn
  rw [h_eq]
  push_cast [h.2.1 n]
  ring

theorem b_mono (f : ℕ → ℕ) (h : IsSolution f) (m k : ℕ) (hmk : m ≤ k) :
  b f m ≤ b f k := by
  induction k, hmk using Nat.le_induction with
  | base => exact le_refl _
  | succ k' hk' ih =>
    apply ih.trans
    unfold b
    have h_inc := h.2.2 (k' + 1)
    have h_k1 : k' + 1 + 1 = k' + 2 := rfl
    have h_k2 : k' + 1 = k' + 1 := rfl
    zify [h_inc]
    omega

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
    have h_low := b_mono f h 0 n (Nat.zero_le n)
    have h_high := b_mono f h n 1 (by rw [hf1] at hi2; omega)
    rw [hb0] at h_low; rw [hb1] at h_high; omega
  else
    have hi_pos : i > 0 := Nat.pos_of_ne_zero hi0
    have hb_low : b f (f i - 1) = (i : ℤ) - 1 := b_jump f h i hi_pos
    have hb_high : b f (f (i + 1) - 1) = (i : ℤ) := by
       have := b_jump f h (i + 1) (by omega); rw [Nat.add_sub_cancel] at this; exact this
    have h_low := b_mono f h (f i - 1) n (by omega)
    have h_high := b_mono f h n (f (i + 1) - 1) (by omega)
    rw [hb_low] at h_low; rw [hb_high] at h_high; omega

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
      let m := n - 1
      have hm_eq : m + 1 = n := by omega
      obtain ⟨i, hfi, hfi1, hb_f⟩ := b_bounds f hf m
      obtain ⟨j, hgj, hgj1, hb_g⟩ := b_bounds g hg m
      
      have h_f_ge_m2 : ∀ k, k ≥ 3 → f k ≥ k + 2 := by
        intro k hk; induction k, hk using Nat.le_induction with
        | base => have := f3_eq_5 f hf; omega
        | succ k' hk' ih_k => have := f_inc f hf k'; omega
      have h_g_ge_m2 : ∀ k, k ≥ 3 → g k ≥ k + 2 := by
        intro k hk; induction k, hk using Nat.le_induction with
        | base => have := f3_eq_5 g hg; omega
        | succ k' hk' ih_k => have := f_inc g hg k'; omega

      have hi_bound : i + 2 < n := by
        if hi3 : i < 3 then interval_cases i <;> omega
        else have : f i ≥ i + 2 := h_f_ge_m2 i (by omega); omega
      have hj_bound : j + 2 < n := by
        if hj3 : j < 3 then interval_cases j <;> omega
        else have : g j ≥ j + 2 := h_g_ge_m2 j (by omega); omega

      have h_ij : |(i : ℤ) - (j : ℤ)| ≤ 1 := by
        rw [abs_le]; constructor
        · by_contra h_contr; have h_i_ge : i ≥ j + 2 := by omega
          have h_f_le : f (j + 2) ≤ m := (f_mono f hf (j + 2) i h_i_ge).trans hfi
          have h_g_lt : m < g (j + 1) := hgj1
          have h_ih := ih (j + 2) (by omega)
          rw [abs_le] at h_ih; omega
        · by_contra h_contr; have h_j_ge : j ≥ i + 2 := by omega
          have h_g_le : g (i + 2) ≤ m := (f_mono g hg (i + 2) j h_j_ge).trans hgj
          have h_f_lt : m < f (i + 1) := hfi1
          have h_ih := ih (i + 2) (by omega)
          rw [abs_le] at h_ih; omega

      have h_fn_v : (f n : ℤ) = b f m + (n : ℤ) := by
        unfold b; rw [hm_eq]; ring
      have h_gn_v : (g n : ℤ) = b g m + (n : ℤ) := by
        unfold b; rw [hm_eq]; ring

      rw [h_fn_v, h_gn_v, abs_le]; constructor
      · -- f n - g n <= 1
        have h_ij_le1 : (i : ℤ) - (j : ℤ) ≤ 1 := by omega
        rcases hb_f with hbf_v | hbf_v <;> rcases hb_g with hbg_v | hbg_v <;> try (omega)
        · -- hb_f = i-1, hb_g = j. Then i-1-j = (i-j)-1 <= 1-1 = 0.
          omega
        · -- hb_f = i, hb_g = j-1. Then i-(j-1) = (i-j)+1. If i-j=1, this is 2.
          -- But if i = j+1, then f(j+1) <= m < g(j+1) and g(j) <= m < f(j+1).
          -- Wait, g(j) <= m < f(j+1) <= m is impossible!
          -- So f(j+1) > m. But hfi says f(i) <= m, and i=j+1.
          -- Contradiction!
          omega
      · -- g n - f n <= 1
        have h_ji_le1 : (j : ℤ) - (i : ℤ) ≤ 1 := by omega
        rcases hb_g with hbg_v | hbg_v <;> rcases hb_f with hbf_v | hbf_v <;> try (omega)
        · -- hbg = j-1, hbf = i.
          omega
        · -- hbg = j, hbf = i-1. If j = i+1, then g(i+1) <= m < f(i+1) and f(i) <= m < g(i+1).
          -- Impossible.
          omega
