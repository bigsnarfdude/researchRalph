import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_422 (x : ℝ) (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 5 * x - 12)
  (h₁ : σ.1 (x + 1) = σ.2 x) : x = 47 / 24 := by
  -- σ.2 x = 5*(x+1) - 12 (from h₁ and h₀)
  have h_symm : σ.2 x = 5 * (x + 1) - 12 := by
    have := h₁; rw [h₀] at this; linarith
  -- x = 5 * σ.2 x - 12 (from σ.1(σ.2 x) = x and h₀)
  have h_eq : x = 5 * σ.2 x - 12 := by
    have h_inv : σ.1 (σ.2 x) = x := σ.right_inv x
    rw [h₀] at h_inv; linarith
  -- Substitute: x = 5*(5*(x+1)-12) - 12 = 25x - 47, so 24x = 47
  linarith
