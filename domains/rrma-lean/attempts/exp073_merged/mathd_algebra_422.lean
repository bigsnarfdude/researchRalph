import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_422 (x : ℝ) (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 5 * x - 12)
  (h₁ : σ.1 (x + 1) = σ.2 x) : x = 47 / 24 := by
  have lhs : σ.1 (x + 1) = 5 * x - 7 := by rw [h₀]; ring
  rw [lhs] at h₁
  have key : σ.1 (5 * x - 7) = x := by
    conv_rhs => rw [← σ.right_inv x]
    congr 1
  rw [h₀] at key
  linarith
