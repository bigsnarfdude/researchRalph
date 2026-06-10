import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_451 (σ : Equiv ℝ ℝ) (h₀ : σ.2 (-15) = 0) (h₁ : σ.2 0 = 3) (h₂ : σ.2 3 = 9)
    (h₃ : σ.2 9 = 20) : σ.1 (σ.1 9) = 0 := by
  -- From σ.2 3 = 9: σ.1 9 = 3
  have h_s9 : σ.1 9 = 3 := by
    have := σ.right_inv 3; rw [h₂] at this; exact this
  -- From σ.2 0 = 3: σ.1 3 = 0
  have h_s3 : σ.1 3 = 0 := by
    have := σ.right_inv 0; rw [h₁] at this; exact this
  rw [h_s9, h_s3]
