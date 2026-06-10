import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_393 (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 4 * x ^ 3 + 1) : σ.2 33 = 2 := by
  -- σ.1 2 = 4 * 8 + 1 = 33, so σ.2 33 = σ.2 (σ.1 2) = 2
  have h2 : σ.1 2 = 33 := by simp [h₀]; norm_num
  have := σ.left_inv 2
  rw [← h2]
  exact σ.left_inv 2