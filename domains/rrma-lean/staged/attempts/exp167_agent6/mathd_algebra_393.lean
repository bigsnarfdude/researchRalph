import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_393 (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 4 * x ^ 3 + 1) : σ.2 33 = 2 := by
  have h : σ.1 2 = 33 := by rw [h₀]; norm_num
  have : σ.2 (σ.1 2) = 2 := σ.left_inv 2
  rwa [h] at this
