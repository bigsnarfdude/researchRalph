import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
  have har : a * r = 2 := by have := h₀ 1; simp [pow_succ, pow_zero] at this; linarith [h₁]
  have har3 : a * r ^ 3 = 6 := by have := h₀ 3; linarith [h₂]
  have hr_ne : r ≠ 0 := by intro hr; rw [hr, mul_zero] at har; linarith
  have ha_ne : a ≠ 0 := by intro ha; rw [ha, zero_mul] at har; linarith
  have hr2 : r ^ 2 = 3 := by
    have : a * r * r ^ 2 = a * r ^ 3 := by ring
    rw [har, har3] at this; linarith
  have ha2 : a ^ 2 = 4 / 3 := by
    have : (a * r) ^ 2 = a ^ 2 * r ^ 2 := by ring
    rw [har, hr2] at this; linarith
  have hu0 : u 0 = a := by rw [h₀]; simp
  rw [hu0]
  have hsqrt3_pos : Real.sqrt 3 > 0 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt3_sq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 3)
  have target_sq : (2 / Real.sqrt 3) ^ 2 = 4 / 3 := by rw [div_pow, hsqrt3_sq]; norm_num
  have : a ^ 2 - (2 / Real.sqrt 3) ^ 2 = 0 := by rw [target_sq]; linarith
  have : (a - 2 / Real.sqrt 3) * (a + 2 / Real.sqrt 3) = 0 := by nlinarith [this]
  rcases mul_eq_zero.mp this with h | h
  · left; linarith
  · right; linarith
