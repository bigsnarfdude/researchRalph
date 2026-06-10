import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) :
  abs (a 2009) = 0 := by
  set s := Real.sqrt 3 with hs_def
  have hs_sq : s ^ 2 = 3 := Real.sq_sqrt (by norm_num : (3 : ℝ) ≥ 0)
  have hs_pos : (0 : ℝ) < s := Real.sqrt_pos.mpr (by norm_num : (3 : ℝ) > 0)
  have h1s : a 2 = s / 3 := by
    rw [h₁]; field_simp [Real.sqrt_ne_zero'.mpr (show (3:ℝ) > 0 by norm_num)]
    linarith [hs_sq]
  -- Helper: apply recurrence, substitute known values, prove algebraic identity
  -- a(3) = 2 + s
  have ha3v : a 3 = 2 + s := by
    have r := h₂ 1 (by norm_num); rw [h₀, h1s] at r; rw [r]
    have hd : (1 : ℝ) - 1 * (s / 3) ≠ 0 := by nlinarith [hs_sq]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(4) = -(2 + s)
  have ha4v : a 4 = -(2 + s) := by
    have r := h₂ 2 (by norm_num); rw [h1s, ha3v] at r; rw [r]
    have hd : (1 : ℝ) - s / 3 * (2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(5) = 0 (numerator: (2+s) + (-(2+s)) = 0)
  have ha5v : a 5 = 0 := by
    have r := h₂ 3 (by norm_num); rw [ha3v, ha4v] at r; rw [r]
    rw [show (2 + s) + -(2 + s) = 0 from by ring, zero_div]
  -- a(6) = -(2+s), a(7) = -(2+s) (from a(5)=0)
  have ha6v : a 6 = -(2 + s) := by
    have r := h₂ 4 (by norm_num); rw [ha4v, ha5v] at r
    simp [mul_zero, sub_zero, div_one, add_zero] at r; linarith
  have ha7v : a 7 = -(2 + s) := by
    have r := h₂ 5 (by norm_num); rw [ha5v, ha6v] at r
    simp [zero_mul, sub_zero, div_one, zero_add] at r; linarith
  -- a(8) = s/3
  have ha8v : a 8 = s / 3 := by
    have r := h₂ 6 (by norm_num); rw [ha6v, ha7v] at r; rw [r]
    have hd : (1 : ℝ) - -(2 + s) * -(2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(9) = -1
  have ha9v : a 9 = -1 := by
    have r := h₂ 7 (by norm_num); rw [ha7v, ha8v] at r; rw [r]
    have hd : (1 : ℝ) - -(2 + s) * (s / 3) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(10) = s - 2
  have ha10v : a 10 = s - 2 := by
    have r := h₂ 8 (by norm_num); rw [ha8v, ha9v] at r; rw [r]
    have hd : (1 : ℝ) - s / 3 * -1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(11) = -s
  have ha11v : a 11 = -s := by
    have r := h₂ 9 (by norm_num); rw [ha9v, ha10v] at r; rw [r]
    have hd : (1 : ℝ) - -1 * (s - 2) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(12) = -(2 + s)
  have ha12v : a 12 = -(2 + s) := by
    have r := h₂ 10 (by norm_num); rw [ha10v, ha11v] at r; rw [r]
    have hd : (1 : ℝ) - (s - 2) * -s ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(13) = 1
  have ha13v : a 13 = 1 := by
    have r := h₂ 11 (by norm_num); rw [ha11v, ha12v] at r; rw [r]
    have hd : (1 : ℝ) - -s * -(2 + s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(14) = -(s/3)
  have ha14v : a 14 = -(s / 3) := by
    have r := h₂ 12 (by norm_num); rw [ha12v, ha13v] at r; rw [r]
    have hd : (1 : ℝ) - -(2 + s) * 1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(15) = 2 - s
  have ha15v : a 15 = 2 - s := by
    have r := h₂ 13 (by norm_num); rw [ha13v, ha14v] at r; rw [r]
    have hd : (1 : ℝ) - 1 * -(s / 3) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(16) = s - 2
  have ha16v : a 16 = s - 2 := by
    have r := h₂ 14 (by norm_num); rw [ha14v, ha15v] at r; rw [r]
    have hd : (1 : ℝ) - -(s / 3) * (2 - s) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(17) = 0 (numerator: (2-s) + (s-2) = 0)
  have ha17v : a 17 = 0 := by
    have r := h₂ 15 (by norm_num); rw [ha15v, ha16v] at r; rw [r]
    rw [show (2 - s) + (s - 2) = 0 from by ring, zero_div]
  -- a(18) = s-2, a(19) = s-2 (from a(17)=0)
  have ha18v : a 18 = s - 2 := by
    have r := h₂ 16 (by norm_num); rw [ha16v, ha17v] at r
    simp [mul_zero, sub_zero, div_one, add_zero] at r; linarith
  have ha19v : a 19 = s - 2 := by
    have r := h₂ 17 (by norm_num); rw [ha17v, ha18v] at r
    simp [zero_mul, sub_zero, div_one, zero_add] at r; linarith
  -- a(20) = -(s/3)
  have ha20v : a 20 = -(s / 3) := by
    have r := h₂ 18 (by norm_num); rw [ha18v, ha19v] at r; rw [r]
    have hd : (1 : ℝ) - (s - 2) * (s - 2) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(21) = -1
  have ha21v : a 21 = -1 := by
    have r := h₂ 19 (by norm_num); rw [ha19v, ha20v] at r; rw [r]
    have hd : (1 : ℝ) - (s - 2) * -(s / 3) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(22) = -(2+s)
  have ha22v : a 22 = -(2 + s) := by
    have r := h₂ 20 (by norm_num); rw [ha20v, ha21v] at r; rw [r]
    have hd : (1 : ℝ) - -(s / 3) * -1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(23) = s
  have ha23v : a 23 = s := by
    have r := h₂ 21 (by norm_num); rw [ha21v, ha22v] at r; rw [r]
    have hd : (1 : ℝ) - -1 * -(2 + s) ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(24) = s - 2
  have ha24v : a 24 = s - 2 := by
    have r := h₂ 22 (by norm_num); rw [ha22v, ha23v] at r; rw [r]
    have hd : (1 : ℝ) - -(2 + s) * s ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(25) = 1
  have ha25v : a 25 = 1 := by
    have r := h₂ 23 (by norm_num); rw [ha23v, ha24v] at r; rw [r]
    have hd : (1 : ℝ) - s * (s - 2) ≠ 0 := by nlinarith [hs_sq, hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- a(26) = s/3
  have ha26v : a 26 = s / 3 := by
    have r := h₂ 24 (by norm_num); rw [ha24v, ha25v] at r; rw [r]
    have hd : (1 : ℝ) - (s - 2) * 1 ≠ 0 := by nlinarith [hs_pos]
    rw [div_eq_iff hd]; nlinarith [hs_sq]
  -- Period 24: a(25)=a(1), a(26)=a(2)
  have h25eq : a 25 = a 1 := by rw [ha25v, h₀]
  have h26eq : a 26 = a 2 := by rw [ha26v, h1s]
  -- Periodicity step lemma
  have hstep : ∀ m, 1 ≤ m → a (m + 24) = a m → a (m + 1 + 24) = a (m + 1) →
      a (m + 2 + 24) = a (m + 2) := by
    intro m hm hm0 hm1
    have r1 := h₂ (m + 24) (by omega)
    have r2 := h₂ m hm
    rw [show m + 24 + 2 = m + 2 + 24 from by omega,
        show m + 24 + 1 = m + 1 + 24 from by omega] at r1
    rw [r1, hm0, hm1, ← r2]
  -- Periodicity by paired induction
  have hperiod : ∀ n, 1 ≤ n → a (n + 24) = a n := by
    suffices ∀ n, 1 ≤ n → a (n + 24) = a n ∧ a (n + 1 + 24) = a (n + 1) by
      exact fun n hn => (this n hn).1
    intro n hn; induction n with
    | zero => omega
    | succ m ih =>
      match m with
      | 0 => exact ⟨h25eq, h26eq⟩
      | m + 1 =>
        obtain ⟨ihm1, ihm2⟩ := ih (by omega)
        exact ⟨ihm2, hstep (m + 1) (by omega) ihm1 ihm2⟩
  -- Extend to multiples
  have hmod : ∀ k n, 1 ≤ n → a (n + 24 * k) = a n := by
    intro k; induction k with
    | zero => intro n _; simp
    | succ k ih =>
      intro n hn
      rw [show n + 24 * (k + 1) = (n + 24 * k) + 24 from by ring,
          hperiod (n + 24 * k) (by omega), ih n hn]
  -- 2009 = 17 + 24*83, a(17) = 0
  rw [show (2009 : ℕ) = 17 + 24 * 83 from by norm_num,
      hmod 83 17 (by norm_num), ha17v, abs_zero]
