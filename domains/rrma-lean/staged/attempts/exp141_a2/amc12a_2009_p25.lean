import Mathlib

set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) : abs (a 2009) = 0 := by
  set s := Real.sqrt 3 with hs_def
  have hs_pos : 0 < s := Real.sqrt_pos.mpr (by norm_num : (3:ℝ) > 0)
  have hs_ne : s ≠ 0 := ne_of_gt hs_pos
  have hs2 : s * s = 3 := Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)
  have hs_gt1 : 1 < s := by nlinarith [hs2]
  -- Macro for recurrence step
  -- a(3) = 2+s
  have ha3 : a 3 = 2 + s := by
    have h3 := h₂ 1 le_rfl; rw [show (1:ℕ)+2=3 from rfl, h₀, h₁] at h3; rw [h3, one_mul]
    have hd : (1:ℝ) - 1 / s ≠ 0 := by intro h; field_simp at h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(4) = -(2+s)
  have ha4 : a 4 = -(2 + s) := by
    have h4 := h₂ 2 (by norm_num); rw [show (2:ℕ)+2=4 from rfl, h₁, ha3] at h4; rw [h4]
    have hd : (1:ℝ) - 1 / s * (2 + s) ≠ 0 := by intro h; field_simp at h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(5) = 0
  have ha5 : a 5 = 0 := by
    have := h₂ 3 (by norm_num); rw [show (3:ℕ)+2=5 from rfl, ha3, ha4] at this; rw [this]; ring_nf
  -- a(6) = -(2+s)
  have ha6 : a 6 = -(2 + s) := by
    have := h₂ 4 (by norm_num); rw [show (4:ℕ)+2=6 from rfl, ha4, ha5] at this; rw [this]; ring_nf
  -- a(7) = -(2+s)
  have ha7 : a 7 = -(2 + s) := by
    have := h₂ 5 (by norm_num); rw [show (5:ℕ)+2=7 from rfl, ha5, ha6] at this; rw [this]; ring_nf
  -- a(8) = 1/s
  have ha8 : a 8 = 1 / s := by
    have := h₂ 6 (by norm_num); rw [show (6:ℕ)+2=8 from rfl, ha6, ha7] at this; rw [this]
    have hd : (1:ℝ) - -(2 + s) * -(2 + s) ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(9) = -1
  have ha9 : a 9 = -1 := by
    have := h₂ 7 (by norm_num); rw [show (7:ℕ)+2=9 from rfl, ha7, ha8] at this; rw [this]
    have hd : (1:ℝ) - -(2 + s) * (1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(10) = s-2
  have ha10 : a 10 = s - 2 := by
    have := h₂ 8 (by norm_num); rw [show (8:ℕ)+2=10 from rfl, ha8, ha9] at this; rw [this]
    have hd : (1:ℝ) - 1 / s * -1 ≠ 0 := by intro h; field_simp at h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(11) = -s
  have ha11 : a 11 = -s := by
    have := h₂ 9 (by norm_num); rw [show (9:ℕ)+2=11 from rfl, ha9, ha10] at this; rw [this]
    have hd : (1:ℝ) - -1 * (s - 2) ≠ 0 := by intro h; nlinarith
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  -- a(12) = -(2+s)
  have ha12 : a 12 = -(2 + s) := by
    have := h₂ 10 (by norm_num); rw [show (10:ℕ)+2=12 from rfl, ha10, ha11] at this; rw [this]
    have hd : (1:ℝ) - (s - 2) * -s ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  -- a(13) = 1
  have ha13 : a 13 = 1 := by
    have := h₂ 11 (by norm_num); rw [show (11:ℕ)+2=13 from rfl, ha11, ha12] at this; rw [this]
    have hd : (1:ℝ) - -s * -(2 + s) ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  -- a(14) = -(1/s)
  have ha14 : a 14 = -(1 / s) := by
    have := h₂ 12 (by norm_num); rw [show (12:ℕ)+2=14 from rfl, ha12, ha13] at this; rw [this]
    have hd : (1:ℝ) - -(2 + s) * 1 ≠ 0 := by intro h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(15) = 2-s
  have ha15 : a 15 = 2 - s := by
    have := h₂ 13 (by norm_num); rw [show (13:ℕ)+2=15 from rfl, ha13, ha14] at this; rw [this]
    have hd : (1:ℝ) - 1 * -(1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(16) = s-2
  have ha16 : a 16 = s - 2 := by
    have := h₂ 14 (by norm_num); rw [show (14:ℕ)+2=16 from rfl, ha14, ha15] at this; rw [this]
    have hd : (1:ℝ) - -(1 / s) * (2 - s) ≠ 0 := by intro h; field_simp at h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- a(17) = 0 (KEY: 2-s + s-2 = 0)
  have ha17 : a 17 = 0 := by
    have := h₂ 15 (by norm_num); rw [show (15:ℕ)+2=17 from rfl, ha15, ha16] at this; rw [this]; ring_nf
  -- Continue for period proof
  have ha18 : a 18 = s - 2 := by
    have := h₂ 16 (by norm_num); rw [show (16:ℕ)+2=18 from rfl, ha16, ha17] at this; rw [this]; ring_nf
  have ha19 : a 19 = s - 2 := by
    have := h₂ 17 (by norm_num); rw [show (17:ℕ)+2=19 from rfl, ha17, ha18] at this; rw [this]; ring_nf
  have ha20 : a 20 = -(1 / s) := by
    have := h₂ 18 (by norm_num); rw [show (18:ℕ)+2=20 from rfl, ha18, ha19] at this; rw [this]
    have hd : (1:ℝ) - (s - 2) * (s - 2) ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  have ha21 : a 21 = -1 := by
    have := h₂ 19 (by norm_num); rw [show (19:ℕ)+2=21 from rfl, ha19, ha20] at this; rw [this]
    have hd : (1:ℝ) - (s - 2) * -(1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith [hs2]
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  have ha22 : a 22 = -(2 + s) := by
    have := h₂ 20 (by norm_num); rw [show (20:ℕ)+2=22 from rfl, ha20, ha21] at this; rw [this]
    have hd : (1:ℝ) - -(1 / s) * -1 ≠ 0 := by intro h; field_simp at h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  have ha23 : a 23 = s := by
    have := h₂ 21 (by norm_num); rw [show (21:ℕ)+2=23 from rfl, ha21, ha22] at this; rw [this]
    have hd : (1:ℝ) - -1 * -(2 + s) ≠ 0 := by intro h; nlinarith
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  have ha24 : a 24 = s - 2 := by
    have := h₂ 22 (by norm_num); rw [show (22:ℕ)+2=24 from rfl, ha22, ha23] at this; rw [this]
    have hd : (1:ℝ) - -(2 + s) * s ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  have ha25 : a 25 = 1 := by
    have := h₂ 23 (by norm_num); rw [show (23:ℕ)+2=25 from rfl, ha23, ha24] at this; rw [this]
    have hd : (1:ℝ) - s * (s - 2) ≠ 0 := by intro h; nlinarith [hs2]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs2]
  have ha26 : a 26 = 1 / s := by
    have := h₂ 24 (by norm_num); rw [show (24:ℕ)+2=26 from rfl, ha24, ha25] at this; rw [this]
    have hd : (1:ℝ) - (s - 2) * 1 ≠ 0 := by intro h; nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hs2]
  -- Period 24: a(n+24) = a(n) for n ≥ 1
  have hperiod : ∀ k, 1 ≤ k → a (k + 24) = a k := by
    suffices ∀ j, ∀ i, 1 ≤ i → i ≤ j + 2 → a (i + 24) = a i by
      intro k hk; exact this (k - 1 + 1) k hk (by omega)
    intro j; induction j with
    | zero =>
      intro i hi1 hi2; interval_cases i
      · exact ha25 ▸ h₀ ▸ rfl
      · exact ha26 ▸ h₁ ▸ rfl
    | succ n ih =>
      intro i hi1 hi2
      by_cases h : i ≤ n + 2
      · exact ih i hi1 h
      · have hi : i = n + 3 := by omega
        subst hi
        have eq1 := ih (n + 1) (by omega) (by omega)
        have eq2 := ih (n + 2) (by omega) (by omega)
        have lhs := h₂ (n + 25) (by omega)
        rw [show n + 25 + 2 = (n + 3) + 24 from by omega,
            show n + 25 = (n + 1) + 24 from by omega,
            show n + 25 + 1 = (n + 2) + 24 from by omega] at lhs
        rw [eq1, eq2] at lhs
        have rhs := h₂ (n + 1) (by omega)
        rw [show n + 1 + 2 = n + 3 from by omega] at rhs
        linarith [lhs, rhs]
  -- 2009 = 17 + 24*83
  have h2009 : a 2009 = a 17 := by
    suffices h : ∀ k : ℕ, a (17 + 24 * k) = a 17 by
      have heq : (2009 : ℕ) = 17 + 24 * 83 := by norm_num
      rw [heq]; exact h 83
    intro k; induction k with
    | zero => simp
    | succ n ih =>
      rw [show 17 + 24 * (n + 1) = (17 + 24 * n) + 24 from by ring]
      rw [hperiod _ (by omega)]; exact ih
  rw [h2009, ha17, abs_zero]
