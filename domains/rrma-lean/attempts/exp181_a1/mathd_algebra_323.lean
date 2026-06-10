import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_323 (σ : Equiv ℝ ℝ) (h : ∀ x, σ.1 x = x ^ 3 - 8) : σ.2 (σ.1 (σ.2 19)) = 3 := by
  -- σ.2 (σ.1 y) = y for all y, so σ.2 (σ.1 (σ.2 19)) = σ.2 19
  -- σ.1 3 = 3^3 - 8 = 19, so σ.2 19 = 3
  have step1 : σ.2 (σ.1 (σ.2 19)) = σ.2 19 := σ.left_inv (σ.2 19)
  have step2 : σ.1 3 = 19 := by simp [h]; norm_num
  rw [step1]
  rw [← step2]
  exact σ.left_inv 3