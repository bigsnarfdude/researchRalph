import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by
  have hprod : (x + 1) * (y + 1) = 81 := by ring_nf; linarith
  have hy_bound : y ≤ 7 := by nlinarith
  interval_cases y <;> omega
