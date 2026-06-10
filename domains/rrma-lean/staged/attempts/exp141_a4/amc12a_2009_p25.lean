import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) :
    abs (a 2009) = 0 := by
  set s := Real.sqrt 3 with hs_def
  have hs_pos : (0 : ℝ) < s := Real.sqrt_pos.mpr (by norm_num : (3:ℝ) > 0)
  have hs2 : s ^ 2 = 3 := Real.sq_sqrt (by norm_num : (3:ℝ) ≥ 0)
  have hs_ne : s ≠ 0 := ne_of_gt hs_pos
  have hss : s * s = 3 := by nlinarith [hs2]
  -- Step-by-step computation: a(3) through a(26)
  have ha3 : a 3 = 2 + s := by
    have h := h₂ 1 (by omega); rw [show (1:ℕ)+2=3 from rfl, show (1:ℕ)+1=2 from rfl] at h
    rw [h₀, h₁] at h; rw [h]
    have : (1 + 1 / s) / (1 - 1 * (1 / s)) = (s + 1) / (s - 1) := by field_simp
    rw [this, div_eq_iff (show s - 1 ≠ 0 by nlinarith [hs2])]; nlinarith [hss]
  have ha4 : a 4 = -(2 + s) := by
    have h := h₂ 2 (by omega); rw [show (2:ℕ)+2=4 from rfl, show (2:ℕ)+1=3 from rfl] at h
    rw [h₁, ha3] at h; rw [h]
    have hd : 1 - 1 / s * (2 + s) ≠ 0 := by
      have : 1 - 1 / s * (2 + s) = -2 / s := by field_simp; ring
      rw [this]; exact div_ne_zero (by norm_num) hs_ne
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha5 : a 5 = 0 := by
    have h := h₂ 3 (by omega); rw [show (3:ℕ)+2=5 from rfl, show (3:ℕ)+1=4 from rfl] at h
    rw [ha3, ha4] at h; rw [h]
    rw [show (2 + s + -(2 + s)) = (0:ℝ) from by ring, zero_div]
  have ha6 : a 6 = -(2 + s) := by
    have h := h₂ 4 (by omega); rw [show (4:ℕ)+2=6 from rfl, show (4:ℕ)+1=5 from rfl] at h
    rw [ha4, ha5] at h; rw [h]; ring
  have ha7 : a 7 = -(2 + s) := by
    have h := h₂ 5 (by omega); rw [show (5:ℕ)+2=7 from rfl, show (5:ℕ)+1=6 from rfl] at h
    rw [ha5, ha6] at h; rw [h]; ring
  have ha8 : a 8 = 1 / s := by
    have h := h₂ 6 (by omega); rw [show (6:ℕ)+2=8 from rfl, show (6:ℕ)+1=7 from rfl] at h
    rw [ha6, ha7] at h; rw [h]
    have hd : 1 - -(2 + s) * -(2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha9 : a 9 = -1 := by
    have h := h₂ 7 (by omega); rw [show (7:ℕ)+2=9 from rfl, show (7:ℕ)+1=8 from rfl] at h
    rw [ha7, ha8] at h; rw [h]
    have hd : 1 - -(2 + s) * (1 / s) ≠ 0 := by
      have : 1 - -(2 + s) * (1 / s) = (2*s+2)/s := by field_simp; ring
      rw [this]; positivity
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha10 : a 10 = -2 + s := by
    have h := h₂ 8 (by omega); rw [show (8:ℕ)+2=10 from rfl, show (8:ℕ)+1=9 from rfl] at h
    rw [ha8, ha9] at h; rw [h]
    have hd : 1 - 1 / s * -1 ≠ 0 := by
      have : 1 - 1 / s * -1 = (s+1)/s := by field_simp; ring
      rw [this]; positivity
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha11 : a 11 = -s := by
    have h := h₂ 9 (by omega); rw [show (9:ℕ)+2=11 from rfl, show (9:ℕ)+1=10 from rfl] at h
    rw [ha9, ha10] at h; rw [h]
    have hd : 1 - -1 * (-2 + s) ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha12 : a 12 = -(2 + s) := by
    have h := h₂ 10 (by omega); rw [show (10:ℕ)+2=12 from rfl, show (10:ℕ)+1=11 from rfl] at h
    rw [ha10, ha11] at h; rw [h]
    have hd : 1 - (-2 + s) * -s ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha13 : a 13 = 1 := by
    have h := h₂ 11 (by omega); rw [show (11:ℕ)+2=13 from rfl, show (11:ℕ)+1=12 from rfl] at h
    rw [ha11, ha12] at h; rw [h]
    have hd : 1 - -s * -(2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha14 : a 14 = -(1 / s) := by
    have h := h₂ 12 (by omega); rw [show (12:ℕ)+2=14 from rfl, show (12:ℕ)+1=13 from rfl] at h
    rw [ha12, ha13] at h; rw [h]
    have hd : 1 - -(2 + s) * 1 ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha15 : a 15 = 2 - s := by
    have h := h₂ 13 (by omega); rw [show (13:ℕ)+2=15 from rfl, show (13:ℕ)+1=14 from rfl] at h
    rw [ha13, ha14] at h; rw [h]
    have hd : 1 - 1 * -(1 / s) ≠ 0 := by
      have : 1 - 1 * -(1 / s) = (s+1)/s := by field_simp; ring
      rw [this]; positivity
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha16 : a 16 = -2 + s := by
    have h := h₂ 14 (by omega); rw [show (14:ℕ)+2=16 from rfl, show (14:ℕ)+1=15 from rfl] at h
    rw [ha14, ha15] at h; rw [h]
    have hd : 1 - -(1 / s) * (2 - s) ≠ 0 := by
      have : 1 - -(1 / s) * (2 - s) = 2/s := by field_simp; ring
      rw [this]; positivity
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha17 : a 17 = 0 := by
    have h := h₂ 15 (by omega); rw [show (15:ℕ)+2=17 from rfl, show (15:ℕ)+1=16 from rfl] at h
    rw [ha15, ha16] at h; rw [h]
    rw [show (2 - s + (-2 + s)) = (0:ℝ) from by ring, zero_div]
  have ha18 : a 18 = -2 + s := by
    have h := h₂ 16 (by omega); rw [show (16:ℕ)+2=18 from rfl, show (16:ℕ)+1=17 from rfl] at h
    rw [ha16, ha17] at h; rw [h]; ring
  have ha19 : a 19 = -2 + s := by
    have h := h₂ 17 (by omega); rw [show (17:ℕ)+2=19 from rfl, show (17:ℕ)+1=18 from rfl] at h
    rw [ha17, ha18] at h; rw [h]; ring
  have ha20 : a 20 = -(1 / s) := by
    have h := h₂ 18 (by omega); rw [show (18:ℕ)+2=20 from rfl, show (18:ℕ)+1=19 from rfl] at h
    rw [ha18, ha19] at h; rw [h]
    have hd : 1 - (-2 + s) * (-2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha21 : a 21 = -1 := by
    have h := h₂ 19 (by omega); rw [show (19:ℕ)+2=21 from rfl, show (19:ℕ)+1=20 from rfl] at h
    rw [ha19, ha20] at h; rw [h]
    have hd : 1 - (-2 + s) * -(1 / s) ≠ 0 := by
      have : 1 - (-2 + s) * -(1 / s) = (2*s-2)/s := by field_simp; ring
      rw [this]; exact div_ne_zero (by nlinarith) hs_ne
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha22 : a 22 = -(2 + s) := by
    have h := h₂ 20 (by omega); rw [show (20:ℕ)+2=22 from rfl, show (20:ℕ)+1=21 from rfl] at h
    rw [ha20, ha21] at h; rw [h]
    have hd : 1 - -(1 / s) * -1 ≠ 0 := by
      have : 1 - -(1 / s) * -1 = (s-1)/s := by field_simp
      rw [this]; exact div_ne_zero (by nlinarith [hs2]) hs_ne
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  have ha23 : a 23 = s := by
    have h := h₂ 21 (by omega); rw [show (21:ℕ)+2=23 from rfl, show (21:ℕ)+1=22 from rfl] at h
    rw [ha21, ha22] at h; rw [h]
    have hd : 1 - -1 * -(2 + s) ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha24 : a 24 = -2 + s := by
    have h := h₂ 22 (by omega); rw [show (22:ℕ)+2=24 from rfl, show (22:ℕ)+1=23 from rfl] at h
    rw [ha22, ha23] at h; rw [h]
    have hd : 1 - -(2 + s) * s ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha25 : a 25 = 1 := by
    have h := h₂ 23 (by omega); rw [show (23:ℕ)+2=25 from rfl, show (23:ℕ)+1=24 from rfl] at h
    rw [ha23, ha24] at h; rw [h]
    have hd : 1 - s * (-2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hss]
  have ha26 : a 26 = 1 / s := by
    have h := h₂ 24 (by omega); rw [show (24:ℕ)+2=26 from rfl, show (24:ℕ)+1=25 from rfl] at h
    rw [ha24, ha25] at h; rw [h]
    have hd : 1 - (-2 + s) * 1 ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; field_simp [hs_ne]; nlinarith [hss]
  -- Period 24: a(n+24) = a(n) for all n ≥ 1
  have hperiod : ∀ n, 1 ≤ n → a (n + 24) = a n := by
    -- Two-step induction: prove ∀ n, P(n+1) ∧ P(n+2) where P(k) = (a(k+24) = a(k))
    suffices h : ∀ n, a (n + 1 + 24) = a (n + 1) ∧ a (n + 2 + 24) = a (n + 2) by
      intro n hn
      obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
      exact (h m).1
    intro n; induction n with
    | zero =>
      constructor
      · show a 25 = a 1; rw [ha25, h₀]
      · show a 26 = a 2; rw [ha26, h₁]
    | succ m ih =>
      obtain ⟨ih1, ih2⟩ := ih
      refine ⟨ih2, ?_⟩
      -- Show a(m+3+24) = a(m+3)
      have eq1 := h₂ (m + 1) (by omega)
      have eq2 := h₂ (m + 1 + 24) (by omega)
      rw [show m + 1 + 24 + 2 = m + 3 + 24 from by omega,
          show m + 1 + 24 + 1 = m + 2 + 24 from by omega,
          show m + 1 + 2 = m + 3 from by omega,
          show m + 1 + 1 = m + 2 from by omega] at *
      rw [ih1, ih2] at eq2
      exact eq2.trans eq1.symm
  -- Final: 2009 ≡ 17 (mod 24), so a(2009) = a(17) = 0
  suffices a 2009 = 0 by rw [this]; simp
  have h2009 : a 2009 = a 17 := by
    suffices ∀ k, a (17 + 24 * k) = a 17 by
      rw [show (2009:ℕ) = 17 + 24 * 83 from by omega]; exact this 83
    intro k; induction k with
    | zero => simp
    | succ m ih =>
      rw [show 17 + 24 * (m + 1) = (17 + 24 * m) + 24 from by ring]
      rw [hperiod (17 + 24 * m) (by omega)]
      exact ih
  rw [h2009, ha17]
