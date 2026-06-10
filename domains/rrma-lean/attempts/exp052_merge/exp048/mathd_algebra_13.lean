import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- Partial fractions: 4x/((x-3)(x-5)) = a/(x-3) + b/(x-5)
-- Plug x=0 and x=1 to get two equations in a,b
theorem mathd_algebra_13 (a b : ℝ)
  (h₀ : ∀ x, x - 3 ≠ 0 ∧ x - 5 ≠ 0 → 4 * x / (x ^ 2 - 8 * x + 15) = a / (x - 3) + b / (x - 5)) :
  a = -6 ∧ b = 10 := by
  have h0 := h₀ 0 (by constructor <;> norm_num)
  have h1 := h₀ 1 (by constructor <;> norm_num)
  -- h0: 0 / 15 = a / (-3) + b / (-5)
  -- h1: 4 / 8 = a / (-2) + b / (-4)
  simp only [zero_mul, zero_div] at h0
  norm_num at h0 h1
  -- h0: a / 3 + b / 5 = 0  (or similar)
  -- h1: ...
  constructor <;> linarith
