import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_89 (b : ℝ) (h₀ : b ≠ 0) :
  (7 * b ^ 3) ^ 2 * (4 * b ^ 2) ^ (-(3 : ℤ)) = 49 / 64 := by
  simp only [zpow_neg]; field_simp; ring
