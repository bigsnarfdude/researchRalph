import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_493 (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 - 4 * Real.sqrt x + 1) :
    f (f 4) = 70 := by
  simp only [h₀]
  rw [show (4:ℝ) = 2^2 from by norm_num, Real.sqrt_sq (by norm_num : (2:ℝ) ≥ 0)]
  norm_num
