import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- x³ - (x+1)(x-1)x = 5 → x = 5 → x³ = 125
-- Simplify: x³ - (x²-1)x = x³ - x³ + x = x = 5
theorem amc12a_2009_p5 (x : ℝ) (h₀ : x ^ 3 - (x + 1) * (x - 1) * x = 5) : x ^ 3 = 125 := by
  have hx : x = 5 := by nlinarith
  rw [hx]; norm_num
