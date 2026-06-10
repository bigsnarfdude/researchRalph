import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12b_2020_p5 (a b : ℕ) (h₀ : (5 : ℚ) / 8 * b = 2 / 3 * a + 7)
  (h₁ : (b : ℚ) - 5 / 8 * b = a - 2 / 3 * a + 7) : a = 42 := by
  have hb : (b : ℚ) = 8 * (2 / 3 * a + 7) / 5 := by linarith
  have : (b : ℚ) * 3 / 8 = (a : ℚ) / 3 + 7 := by linarith
  have : (a : ℚ) = 42 := by linarith
  exact_mod_cast this
