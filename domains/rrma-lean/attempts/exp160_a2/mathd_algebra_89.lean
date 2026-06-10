import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem mathd_algebra_89 (b : ℝ) (h₀ : b ≠ 0) :
  (7 * b ^ 3) ^ 2 * (4 * b ^ 2) ^ (-(3 : ℤ)) = 49 / 64 := by
  have hb : b ≠ 0 := h₀
  have hb6 : b ^ 6 ≠ 0 := pow_ne_zero 6 hb
  rw [show (-(3:ℤ)) = -↑(3:ℕ) from rfl, zpow_neg, zpow_natCast]
  rw [show (4 * b ^ 2) ^ 3 = 64 * b ^ 6 by ring]
  rw [show (7 * b ^ 3) ^ 2 = 49 * b ^ 6 by ring]
  field_simp
