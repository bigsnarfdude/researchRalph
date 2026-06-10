import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- h₀: 5/8 * b = 2/3 * a + 7
-- h₁: b - 5/8 * b = a - 2/3 * a + 7, i.e., 3/8 * b = 1/3 * a + 7
-- From h₀: 5b/8 = 2a/3 + 7 → 15b = 16a + 168
-- From h₁: 3b/8 = a/3 + 7 → 9b = 8a + 168
-- Subtract: 6b = 8a → b = 4a/3
-- Sub back: 9*(4a/3) = 8a + 168 → 12a = 8a + 168 → 4a = 168 → a = 42
theorem amc12b_2020_p5 (a b : ℕ) (h₀ : (5 : ℚ) / 8 * b = 2 / 3 * a + 7)
  (h₁ : (b : ℚ) - 5 / 8 * b = a - 2 / 3 * a + 7) : a = 42 := by
  have h2 : (3 : ℚ) / 8 * b = (1 : ℚ) / 3 * a + 7 := by linarith
  -- From h₀ and h2: multiply h₀ by 3 and h2 by 5: 15b/8 = 2a+21, 15b/8 = 5a/3+35
  -- So 2a+21 = 5a/3+35, 6a+63=5a+105, a=42
  have h3 : (a : ℚ) = 42 := by linarith
  exact_mod_cast h3
