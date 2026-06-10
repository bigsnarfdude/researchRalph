import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
-- Known impossible: Real.log handles negatives, counterexample a=-64,b=8
theorem aime_1984_p5 (a b : ℝ) (h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5)
  (h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7) : a * b = 512 := by
  simp only [Real.logb] at *
  nlinarith [h₀, h₁, sq_nonneg a, sq_nonneg b, Real.log_pos (by norm_num : (1:ℝ) < 2)]
