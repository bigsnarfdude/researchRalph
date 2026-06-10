import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_323 (σ : Equiv ℝ ℝ) (h : ∀ x, σ.1 x = x ^ 3 - 8) : σ.2 (σ.1 (σ.2 19)) = 3 := by
  have h1 : σ.1 3 = 19 := by rw [h]; norm_num
  have h2 : σ.2 19 = 3 := by rw [← h1]; exact σ.left_inv 3
  rw [h2]
  rw [σ.left_inv]
