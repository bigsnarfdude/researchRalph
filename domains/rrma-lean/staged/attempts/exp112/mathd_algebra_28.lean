import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_28 (c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 + 5 * x + c)
  (h₁ : ∃ x, f x ≤ 0) : c ≤ 25 / 8 := by
  obtain ⟨x, hx⟩ := h₁
  rw [h₀] at hx
  nlinarith [sq_nonneg (4*x + 5)]
