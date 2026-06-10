import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2011_p18 (x y : ℝ) (h₀ : abs (x + y) + abs (x - y) = 2) :
  x ^ 2 - 6 * x + y ^ 2 ≤ 8 := by
  -- Case split to remove abs
  rcases le_or_gt 0 (x + y) with hxy | hxy <;>
  rcases le_or_gt 0 (x - y) with hxy' | hxy' <;>
  simp only [abs_of_nonneg, abs_of_neg, abs_of_nonpos, *, le_of_lt] at h₀ <;>
  nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg (x + 1), sq_nonneg (y + 1), sq_nonneg (y - 1)]
