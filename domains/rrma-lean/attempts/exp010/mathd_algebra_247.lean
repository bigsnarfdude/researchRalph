import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_247 (t s : ℝ) (n : ℤ) (h₀ : t = 2 * s - s ^ 2) (h₁ : s = n ^ 2 - 2 ^ n + 1)
  (h₂ : n = 3) : t = 0 := by
  subst h₂; norm_num at h₁; subst h₁; linarith
