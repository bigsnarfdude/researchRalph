import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_421 (a b c d : ℝ) (h₀ : b = a ^ 2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a ^ 2 + a + 6) (h₂ : d = c ^ 2 + 4 * c + 6) (h₃ : d = 1 / 2 * c ^ 2 + c + 6)
  (h₄ : a < c) : c - a = 6 := by
  have ha : a * (a + 6) = 0 := by nlinarith
  have hc : c * (c + 6) = 0 := by nlinarith
  rcases mul_eq_zero.mp ha with ha' | ha' <;> rcases mul_eq_zero.mp hc with hc' | hc' <;> linarith
