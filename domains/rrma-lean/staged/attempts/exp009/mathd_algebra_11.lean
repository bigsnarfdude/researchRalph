import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_11 (a b : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ 2 * b)
    (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) : (a + 11 * b) / (a - b) = 2 := by
  have hab : a - 2 * b ≠ 0 := sub_ne_zero.mpr h₁
  rw [div_eq_iff hab] at h₂
  have h4 : a = 13 * b := by linarith
  have hab2 : a - b ≠ 0 := by intro h; apply h₀; linarith
  rw [div_eq_iff hab2]
  linarith
