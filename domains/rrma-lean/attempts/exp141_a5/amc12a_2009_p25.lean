import Mathlib

set_option maxHeartbeats 128000000

open BigOperators Real Nat Topology Rat

-- Helper lemma 1: a(9) = -1 given s = sqrt(3)
private lemma aux1 (a : ℕ → ℝ)
    (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1)))
    (s : ℝ) (hs_sq : s ^ 2 = 3) (hs_pos : 0 < s)
    (ha1 : a 1 = 1) (ha2 : a 2 = s / 3) :
    a 5 = 0 ∧ a 9 = -1 ∧ a 8 = s / 3 := by
  have ha3 := h₂ 1 (by norm_num)
  have ha4 := h₂ 2 (by norm_num)
  have ha5 := h₂ 3 (by norm_num)
  have ha6 := h₂ 4 (by norm_num)
  have ha7 := h₂ 5 (by norm_num)
  have ha8 := h₂ 6 (by norm_num)
  have ha9 := h₂ 7 (by norm_num)
  have ha3_val : a 3 = 2 + s := by
    rw [ha3, ha1, ha2]
    have hd : (1 : ℝ) - s / 3 ≠ 0 := by nlinarith [hs_sq]
    rw [show (1:ℝ) - 1 * (s/3) = 1 - s/3 from by ring, div_eq_iff hd]; nlinarith [hs_sq]
  have ha4_val : a 4 = -(2 + s) := by
    rw [ha4, ha2, ha3_val]
    have hd : (1 : ℝ) - s / 3 * (2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; ring_nf; nlinarith [hs_sq]
  have ha5_zero : a 5 = 0 := by
    rw [ha5, ha3_val, ha4_val]; ring_nf
  have ha6_val : a 6 = -(2 + s) := by
    rw [ha6, ha5_zero, add_zero, mul_zero, sub_zero, div_one]; exact ha4_val
  have ha7_val : a 7 = -(2 + s) := by
    rw [ha7, ha5_zero, zero_add, zero_mul, sub_zero, div_one]; exact ha6_val
  have ha8_val : a 8 = s / 3 := by
    rw [ha8, ha6_val, ha7_val]
    have hd : (1 : ℝ) - -(2 + s) * -(2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha9_val : a 9 = -1 := by
    rw [ha9, ha7_val, ha8_val]
    have hd : (1 : ℝ) - -(2 + s) * (s / 3) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  exact ⟨ha5_zero, ha9_val, ha8_val⟩

-- Helper lemma 2: a(17) = 0 given a(8) = s/3, a(9) = -1
private lemma aux2 (a : ℕ → ℝ)
    (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1)))
    (s : ℝ) (hs_sq : s ^ 2 = 3) (hs_pos : 0 < s)
    (ha8_val : a 8 = s / 3) (ha9_val : a 9 = -1) :
    a 17 = 0 ∧ a 16 = s - 2 := by
  have ha10 := h₂ 8 (by norm_num)
  have ha11 := h₂ 9 (by norm_num)
  have ha12 := h₂ 10 (by norm_num)
  have ha13 := h₂ 11 (by norm_num)
  have ha14 := h₂ 12 (by norm_num)
  have ha15 := h₂ 13 (by norm_num)
  have ha16 := h₂ 14 (by norm_num)
  have ha17 := h₂ 15 (by norm_num)
  have ha10_val : a 10 = s - 2 := by
    rw [ha10, ha8_val, ha9_val]
    have hd : (1 : ℝ) - s / 3 * -1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha11_val : a 11 = -s := by
    rw [ha11, ha9_val, ha10_val]
    have hd : (1 : ℝ) - -1 * (s - 2) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha12_val : a 12 = -(2 + s) := by
    rw [ha12, ha10_val, ha11_val]
    have hd : (1 : ℝ) - (s - 2) * -s ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha13_val : a 13 = 1 := by
    rw [ha13, ha11_val, ha12_val]
    have hd : (1 : ℝ) - -s * -(2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha14_val : a 14 = -(s / 3) := by
    rw [ha14, ha12_val, ha13_val]
    have hd : (1 : ℝ) - -(2 + s) * 1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha15_val : a 15 = 2 - s := by
    rw [ha15, ha13_val, ha14_val]
    have hd : (1 : ℝ) - 1 * -(s / 3) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha16_val : a 16 = s - 2 := by
    rw [ha16, ha14_val, ha15_val]
    have hd : (1 : ℝ) - -(s / 3) * (2 - s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha17_zero : a 17 = 0 := by
    rw [ha17, ha15_val, ha16_val]; ring_nf
  exact ⟨ha17_zero, ha16_val⟩

-- Helper lemma 3: a(25) = 1 and a(26) = s/3 given a(16) = s-2, a(17) = 0
private lemma aux3 (a : ℕ → ℝ)
    (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1)))
    (s : ℝ) (hs_sq : s ^ 2 = 3) (hs_pos : 0 < s)
    (ha16_val : a 16 = s - 2) (ha17_zero : a 17 = 0) :
    a 25 = 1 ∧ a 26 = s / 3 := by
  have ha18 := h₂ 16 (by norm_num)
  have ha19 := h₂ 17 (by norm_num)
  have ha20 := h₂ 18 (by norm_num)
  have ha21 := h₂ 19 (by norm_num)
  have ha22 := h₂ 20 (by norm_num)
  have ha23 := h₂ 21 (by norm_num)
  have ha24 := h₂ 22 (by norm_num)
  have ha25 := h₂ 23 (by norm_num)
  have ha26 := h₂ 24 (by norm_num)
  have ha18_val : a 18 = s - 2 := by
    rw [ha18, ha17_zero, add_zero, mul_zero, sub_zero, div_one]; exact ha16_val
  have ha19_val : a 19 = s - 2 := by
    rw [ha19, ha17_zero, zero_add, zero_mul, sub_zero, div_one]; exact ha18_val
  have ha20_val : a 20 = -(s / 3) := by
    rw [ha20, ha18_val, ha19_val]
    have hd : (1 : ℝ) - (s - 2) * (s - 2) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha21_val : a 21 = -1 := by
    rw [ha21, ha19_val, ha20_val]
    have hd : (1 : ℝ) - (s - 2) * -(s / 3) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha22_val : a 22 = -(2 + s) := by
    rw [ha22, ha20_val, ha21_val]
    have hd : (1 : ℝ) - -(s / 3) * -1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha23_val : a 23 = s := by
    rw [ha23, ha21_val, ha22_val]
    have hd : (1 : ℝ) - -1 * -(2 + s) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha24_val : a 24 = s - 2 := by
    rw [ha24, ha22_val, ha23_val]
    have hd : (1 : ℝ) - -(2 + s) * s ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha25_val : a 25 = 1 := by
    rw [ha25, ha23_val, ha24_val]
    have hd : (1 : ℝ) - s * (s - 2) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  have ha26_val : a 26 = s / 3 := by
    rw [ha26, ha24_val, ha25_val]
    have hd : (1 : ℝ) - (s - 2) * 1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  exact ⟨ha25_val, ha26_val⟩

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) : abs (a 2009) = 0 := by
  set s := Real.sqrt 3 with hs_def
  have hs_sq : s ^ 2 = 3 := Real.sq_sqrt (by norm_num : (3 : ℝ) ≥ 0)
  have hs_pos : (0 : ℝ) < s := Real.sqrt_pos.mpr (by norm_num : (3 : ℝ) > 0)
  -- a 2 = s/3
  have h1s : a 2 = s / 3 := by
    rw [h₁]
    rw [div_eq_div_iff (Real.sqrt_ne_zero'.mpr (by norm_num : (3:ℝ) > 0)) (by norm_num : (3:ℝ) ≠ 0)]
    nlinarith [hs_sq]
  -- Apply helper lemmas
  obtain ⟨ha5_zero, ha9_val, ha8_val⟩ := aux1 a h₂ s hs_sq hs_pos h₀ h1s
  obtain ⟨ha17_zero, ha16_val⟩ := aux2 a h₂ s hs_sq hs_pos ha8_val ha9_val
  obtain ⟨ha25_val, ha26_val⟩ := aux3 a h₂ s hs_sq hs_pos ha16_val ha17_zero
  -- Period 24: a(25) = a(1) and a(26) = a(2)
  have h25_eq : a 25 = a 1 := by rw [ha25_val, h₀]
  have h26_eq : a 26 = a 2 := by rw [ha26_val, h1s]
  -- Periodicity by strong induction
  have hperiod : ∀ n, 1 ≤ n → a (n + 24) = a n := by
    intro n hn
    induction n using Nat.strong_induction_on with
    | h n ih =>
      match n, hn with
      | 1, _ => exact h25_eq
      | 2, _ => exact h26_eq
      | n + 3, _ =>
        have ih1 := ih (n + 1) (by omega) (by omega)
        have ih2 := ih (n + 2) (by omega) (by omega)
        have rec_s := h₂ (n + 1 + 24) (by omega)
        have rec_o := h₂ (n + 1) (by omega)
        rw [show n + 1 + 24 + 2 = n + 3 + 24 from by omega,
            show n + 1 + 24 + 1 = n + 2 + 24 from by omega] at rec_s
        rw [show n + 1 + 2 = n + 3 from by omega,
            show n + 1 + 1 = n + 2 from by omega] at rec_o
        rw [rec_s, ih1, ih2, ← rec_o]
  -- Extend to multiples of 24
  have hperiod_k : ∀ k n, 1 ≤ n → a (n + 24 * k) = a n := by
    intro k; induction k with
    | zero => intro n _; simp
    | succ k ih =>
      intro n hn
      rw [show n + 24 * (k + 1) = (n + 24 * k) + 24 from by ring]
      rw [hperiod (n + 24 * k) (by omega), ih n hn]
  -- 2009 = 17 + 24 * 83
  rw [show (2009 : ℕ) = 17 + 24 * 83 from by norm_num]
  rw [hperiod_k 83 17 (by norm_num), ha17_zero, abs_zero]
