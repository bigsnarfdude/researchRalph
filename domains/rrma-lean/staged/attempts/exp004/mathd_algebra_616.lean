import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_616 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 3 + 2 * x + 1)
    (h₁ : ∀ x, g x = x - 1) : f (g 1) = 1 := by
  first
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide