import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by
  have hsqrt : Real.sqrt 131 ^ 2 = 131 := by
    rw [Real.sq_sqrt]
    norm_num
  have hk : k = 13 * x - 2 * x ^ 2 := by linarith
  rw [hk, h₀]
  have : (131 : ℝ) ≥ 0 := by norm_num
  nlinarith [Real.sq_sqrt (show (131 : ℝ) ≥ 0 by norm_num), Real.sqrt_nonneg 131,
             sq_nonneg (Real.sqrt 131), sq_nonneg (Real.sqrt 131 - 13)]
