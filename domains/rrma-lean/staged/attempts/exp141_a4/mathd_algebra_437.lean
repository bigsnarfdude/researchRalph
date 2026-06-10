import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_437 (x y : ℝ) (n : ℤ) (h₀ : x ^ 3 = -45) (h₁ : y ^ 3 = -101) (h₂ : x < n)
  (h₃ : ↑n < y) : n = -4 := by
  exfalso
  -- x³ = -45 > -101 = y³, so x > y (cube is monotone)
  -- But h₂, h₃ give x < ↑n < y, so x < y. Contradiction.
  have hlt : x < y := lt_trans h₂ h₃
  have hcubes : x ^ 3 > y ^ 3 := by linarith
  have hyx : y - x > 0 := by linarith
  have hsq : x ^ 2 + x * y + y ^ 2 ≥ 0 := by nlinarith [sq_nonneg (2 * x + y), sq_nonneg y]
  have hfact : y ^ 3 - x ^ 3 = (y - x) * (x ^ 2 + x * y + y ^ 2) := by ring
  linarith [mul_nonneg (le_of_lt hyx) hsq]
