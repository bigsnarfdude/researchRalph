import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_247 (t s : ℝ) (n : ℤ) (h₀ : t = 2 * s - s ^ 2) (h₁ : s = n ^ 2 - 2 ^ n + 1)
  (h₂ : n = 3) : t = 0 := by
  rw [h₂] at h₁; norm_num at h₁; rw [h₁] at h₀; norm_num at h₀; exact h₀
