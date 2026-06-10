import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : y ^ 3 = 1)
  (h₂ : 6 * x ^ 2 = 2 * (6 * y ^ 2)) : x ^ 3 = 2 * Real.sqrt 2 := by
  have hx := h₀.1; have hy := h₀.2
  have hy1 : y = 1 := by nlinarith [sq_nonneg (y - 1), sq_nonneg y]
  have hx2 : x ^ 2 = 2 := by nlinarith
  have hxsqrt : x = Real.sqrt 2 := by rw [← Real.sqrt_sq (le_of_lt hx), hx2]
  rw [hxsqrt, show Real.sqrt 2 ^ 3 = Real.sqrt 2 * (Real.sqrt 2 * Real.sqrt 2) from by ring]
  rw [Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)]; ring
