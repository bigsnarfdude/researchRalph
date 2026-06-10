import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by
  have h3 : (x + 1) * (y + 1) = 81 := by nlinarith
  have hxy : y + 1 < x + 1 := by omega
  have hbound : (y + 1) * (y + 1) < 81 := by nlinarith
  have : y + 1 ≤ 8 := by nlinarith
  have : 1 ≤ y := by omega; have : y ≤ 7 := by omega
  interval_cases y <;> omega
