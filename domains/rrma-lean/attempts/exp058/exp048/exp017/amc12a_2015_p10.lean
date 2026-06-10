import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by
  have h3 : (x + 1) * (y + 1) = 81 := by nlinarith
  have hy_pos : 2 ≤ y + 1 := by omega
  have hx_gt_y : y + 1 < x + 1 := by omega
  -- x + 1 = 81 / (y + 1). Since y + 1 ≥ 2 and divides 81, and y+1 < x+1
  -- try: y+1 must be 3 (only factor of 81 in range [2, 8])
  -- x = 81/(y+1) - 1. Also (y+1)^2 < 81, so y+1 ≤ 8.
  have hyu : y + 1 ≤ 8 := by nlinarith
  -- Now just check all y in [1..7]
  have : y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 := by omega
  rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega
