import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_616 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 3 + 2 * x + 1)
    (h₁ : ∀ x, g x = x - 1) : f (g 1) = 1 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg g, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (f - g), sq_nonneg (f + g), mul_self_nonneg (f - g)]
    | simp_all [*]
    | decide