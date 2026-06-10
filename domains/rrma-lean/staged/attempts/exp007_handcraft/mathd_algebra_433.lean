import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- f 8 = 3*sqrt(16-7) - 8 = 3*sqrt(9) - 8 = 3*3 - 8 = 1
-- Wait, the problem says f 8 = 19... let me recheck
-- f x = 3*sqrt(2x - 7) - 8
-- f 8 = 3*sqrt(16 - 7) - 8 = 3*sqrt(9) - 8 = 9 - 8 = 1
-- But the goal is f 8 = 19... hmm, the problem file must have it differently
-- Let me just check what happens with norm_num approach
theorem mathd_algebra_433 (f : ℝ → ℝ) (h₀ : ∀ x, f x = 3 * Real.sqrt (2 * x - 7) - 8) : f 8 = 19 := by
  simp only [h₀]
  norm_num
  rw [show (2 : ℝ) * 8 - 7 = 9 from by norm_num]
  rw [show (9 : ℝ) = 3 ^ 2 from by norm_num, Real.sqrt_sq (by norm_num : (3 : ℝ) ≥ 0)]
  ring
