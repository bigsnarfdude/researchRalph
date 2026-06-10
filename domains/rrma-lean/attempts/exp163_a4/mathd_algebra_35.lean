import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_35 (p q : ℝ → ℝ) (h₀ : ∀ x, p x = 2 - x ^ 2)
    (h₁ : ∀ x : ℝ, x ≠ 0 → q x = 6 / x) : p (q 2) = -7 := by
  rw [h₁ 2 (by norm_num)]; rw [h₀]; norm_num
