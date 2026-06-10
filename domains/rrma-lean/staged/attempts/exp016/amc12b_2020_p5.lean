import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12b_2020_p5 (a b : ℕ) (h₀ : (5 : ℚ) / 8 * b = 2 / 3 * a + 7)
  (h₁ : (b : ℚ) - 5 / 8 * b = a - 2 / 3 * a + 7) : a = 42 := by
  have h2 : (3 : ℚ) / 8 * b = 1 / 3 * a + 7 := by linarith
  have h3 : (1 : ℚ) / 4 * b = 1 / 3 * a := by linarith
  have h4 : (b : ℚ) = 4 / 3 * a := by linarith
  have h5 : (5 : ℚ) / 8 * (4 / 3 * a) = 2 / 3 * a + 7 := by linarith
  have h6 : (a : ℚ) = 42 := by linarith
  exact_mod_cast h6
