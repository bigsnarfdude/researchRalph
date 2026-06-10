import Mathlib

set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) :
    abs (a 2009) = 0 := by
  set s := Real.sqrt 3 with hs_def
  have hs_pos : (0 : ℝ) < s := Real.sqrt_pos_of_pos (by norm_num)
  have hs_ne : s ≠ 0 := ne_of_gt hs_pos
  have hss : s * s = 3 := Real.mul_self_sqrt (by norm_num)
  have hs_gt1 : 1 < s := by nlinarith [hss]
  -- a(3) = 2 + s
  have ha3 : a 3 = 2 + s := by
    have h := h₂ 1 le_rfl
    simp only [show (1:ℕ)+2=3 from rfl, show (1:ℕ)+1=2 from rfl] at h
    rw [h, h₀, h₁]
    have hd : 1 - 1 * (1 / s) ≠ 0 := by intro h; field_simp at h; linarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(4) = -(2 + s)
  have ha4 : a 4 = -(2 + s) := by
    have h := h₂ 2 (by norm_num)
    simp only [show (2:ℕ)+2=4 from rfl, show (2:ℕ)+1=3 from rfl] at h
    rw [h, h₁, ha3]
    have hd : 1 - 1 / s * (2 + s) ≠ 0 := by
      intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(5) = 0
  have ha5 : a 5 = 0 := by
    have h := h₂ 3 (by norm_num)
    simp only [show (3:ℕ)+2=5 from rfl, show (3:ℕ)+1=4 from rfl] at h
    rw [h, ha3, ha4]; show ((2 + s) + -(2 + s)) / _ = 0; rw [show (2 : ℝ) + s + -(2 + s) = 0 from by ring]; simp
  -- a(6), a(7) via a(5)=0
  have ha6 : a 6 = -(2 + s) := by
    have h := h₂ 4 (by norm_num)
    simp only [show (4:ℕ)+2=6 from rfl, show (4:ℕ)+1=5 from rfl] at h
    rw [h, ha4, ha5]; simp
  have ha7 : a 7 = -(2 + s) := by
    have h := h₂ 5 (by norm_num)
    simp only [show (5:ℕ)+2=7 from rfl, show (5:ℕ)+1=6 from rfl] at h
    rw [h, ha5, ha6]; simp
  -- a(8) = 1/s
  have ha8 : a 8 = 1 / s := by
    have h := h₂ 6 (by norm_num)
    simp only [show (6:ℕ)+2=8 from rfl, show (6:ℕ)+1=7 from rfl] at h
    rw [h, ha6, ha7]
    have hd : 1 - -(2 + s) * -(2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(9) = -1
  have ha9 : a 9 = -1 := by
    have h := h₂ 7 (by norm_num)
    simp only [show (7:ℕ)+2=9 from rfl, show (7:ℕ)+1=8 from rfl] at h
    rw [h, ha7, ha8]
    have hd : 1 - -(2 + s) * (1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(10) = s - 2
  have ha10 : a 10 = s - 2 := by
    have h := h₂ 8 (by norm_num)
    simp only [show (8:ℕ)+2=10 from rfl, show (8:ℕ)+1=9 from rfl] at h
    rw [h, ha8, ha9]
    have hd : 1 - 1 / s * (-1) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(11) = -s
  have ha11 : a 11 = -s := by
    have h := h₂ 9 (by norm_num)
    simp only [show (9:ℕ)+2=11 from rfl, show (9:ℕ)+1=10 from rfl] at h
    rw [h, ha9, ha10]
    have hd : 1 - (-1) * (s - 2) ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; nlinarith [hss]
  -- a(12) = -(2+s)
  have ha12 : a 12 = -(2 + s) := by
    have h := h₂ 10 (by norm_num)
    simp only [show (10:ℕ)+2=12 from rfl, show (10:ℕ)+1=11 from rfl] at h
    rw [h, ha10, ha11]
    have hd : 1 - (s - 2) * (-s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; nlinarith [hss]
  -- a(13) = 1
  have ha13 : a 13 = 1 := by
    have h := h₂ 11 (by norm_num)
    simp only [show (11:ℕ)+2=13 from rfl, show (11:ℕ)+1=12 from rfl] at h
    rw [h, ha11, ha12]
    have hd : 1 - (-s) * -(2 + s) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; nlinarith [hss]
  -- a(14) = -1/s
  have ha14 : a 14 = -(1 / s) := by
    have h := h₂ 12 (by norm_num)
    simp only [show (12:ℕ)+2=14 from rfl, show (12:ℕ)+1=13 from rfl] at h
    rw [h, ha12, ha13]
    have hd : 1 - -(2 + s) * 1 ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(15) = 2 - s
  have ha15 : a 15 = 2 - s := by
    have h := h₂ 13 (by norm_num)
    simp only [show (13:ℕ)+2=15 from rfl, show (13:ℕ)+1=14 from rfl] at h
    rw [h, ha13, ha14]
    have hd : 1 - 1 * -(1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(16) = s - 2
  have ha16 : a 16 = s - 2 := by
    have h := h₂ 14 (by norm_num)
    simp only [show (14:ℕ)+2=16 from rfl, show (14:ℕ)+1=15 from rfl] at h
    rw [h, ha14, ha15]
    have hd : 1 - -(1 / s) * (2 - s) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- a(17) = 0 (since a(15)+a(16) = (2-s)+(s-2) = 0)
  have ha17 : a 17 = 0 := by
    have h := h₂ 15 (by norm_num)
    simp only [show (15:ℕ)+2=17 from rfl, show (15:ℕ)+1=16 from rfl] at h
    rw [h, ha15, ha16]; simp
  -- a(18)..a(26) for period 24
  have ha18 : a 18 = s - 2 := by
    have h := h₂ 16 (by norm_num)
    simp only [show (16:ℕ)+2=18 from rfl, show (16:ℕ)+1=17 from rfl] at h
    rw [h, ha16, ha17]; simp
  have ha19 : a 19 = s - 2 := by
    have h := h₂ 17 (by norm_num)
    simp only [show (17:ℕ)+2=19 from rfl, show (17:ℕ)+1=18 from rfl] at h
    rw [h, ha17, ha18]; simp
  have ha20 : a 20 = -(1 / s) := by
    have h := h₂ 18 (by norm_num)
    simp only [show (18:ℕ)+2=20 from rfl, show (18:ℕ)+1=19 from rfl] at h
    rw [h, ha18, ha19]
    have hd : 1 - (s - 2) * (s - 2) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  have ha21 : a 21 = -1 := by
    have h := h₂ 19 (by norm_num)
    simp only [show (19:ℕ)+2=21 from rfl, show (19:ℕ)+1=20 from rfl] at h
    rw [h, ha19, ha20]
    have hd : 1 - (s - 2) * -(1 / s) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  have ha22 : a 22 = -(2 + s) := by
    have h := h₂ 20 (by norm_num)
    simp only [show (20:ℕ)+2=22 from rfl, show (20:ℕ)+1=21 from rfl] at h
    rw [h, ha20, ha21]
    have hd : 1 - -(1 / s) * (-1) ≠ 0 := by intro h; field_simp at h; nlinarith [hss]
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  have ha23 : a 23 = s := by
    have h := h₂ 21 (by norm_num)
    simp only [show (21:ℕ)+2=23 from rfl, show (21:ℕ)+1=22 from rfl] at h
    rw [h, ha21, ha22]
    have hd : 1 - (-1) * -(2 + s) ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; nlinarith [hss]
  have ha24 : a 24 = s - 2 := by
    have h := h₂ 22 (by norm_num)
    simp only [show (22:ℕ)+2=24 from rfl, show (22:ℕ)+1=23 from rfl] at h
    rw [h, ha22, ha23]
    have hd : 1 - -(2 + s) * s ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; nlinarith [hss]
  have ha25 : a 25 = 1 := by
    have h := h₂ 23 (by norm_num)
    simp only [show (23:ℕ)+2=25 from rfl, show (23:ℕ)+1=24 from rfl] at h
    rw [h, ha23, ha24]
    have hd : 1 - s * (s - 2) ≠ 0 := by nlinarith [hss]
    rw [div_eq_iff hd]; nlinarith [hss]
  have ha26 : a 26 = 1 / s := by
    have h := h₂ 24 (by norm_num)
    simp only [show (24:ℕ)+2=26 from rfl, show (24:ℕ)+1=25 from rfl] at h
    rw [h, ha24, ha25]
    have hd : 1 - (s - 2) * 1 ≠ 0 := by nlinarith
    rw [div_eq_iff hd]; field_simp; nlinarith [hss]
  -- Period 24
  have hperiod : ∀ n, 1 ≤ n → a (n + 24) = a n := by
    have base1 : a 25 = a 1 := by linarith [ha25, h₀]
    have base2 : a 26 = a 2 := by linarith [ha26, h₁]
    have pairs : ∀ k, a (k + 25) = a (k + 1) ∧ a (k + 26) = a (k + 2) := by
      intro k; induction k with
      | zero => exact ⟨by simpa using base1, by simpa using base2⟩
      | succ m ih =>
        refine ⟨ih.2, ?_⟩
        have hrec := h₂ (m + 1) (by omega)
        rw [show m + 1 + 2 = m + 3 from by omega] at hrec
        have hrec24 := h₂ (m + 25) (by omega)
        rw [show m + 25 + 2 = m + 27 from by omega, show m + 25 + 1 = m + 26 from by omega,
            show (m + 25 : ℕ) = m + 1 + 24 from by omega] at hrec24
        rw [show m + 1 + 24 = m + 25 from by omega] at hrec24
        rw [show m + 1 + 1 + 24 = m + 26 from by omega] at *
        rw [hrec24, ih.1, ih.2]; linarith [hrec]
    intro n hn; rcases n with _ | m; omega
    rw [show m + 1 + 24 = m + 25 from by omega]; exact (pairs m).1
  -- 2009 = 17 + 24*83, so a(2009) = a(17) = 0
  have ha2009 : a 2009 = 0 := by
    have iter : ∀ m, a (17 + 24 * m) = a 17 := by
      intro m; induction m with
      | zero => simp
      | succ k ih =>
        rw [show 17 + 24 * (k + 1) = (17 + 24 * k) + 24 from by ring]
        exact (hperiod _ (by omega)).trans ih
    rw [show (2009 : ℕ) = 17 + 24 * 83 from by norm_num]; rw [iter 83, ha17]
  rw [ha2009, abs_zero]
