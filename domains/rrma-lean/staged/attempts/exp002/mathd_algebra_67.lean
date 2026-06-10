import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_67 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = 5 * x + 3) (h₁ : ∀ x, g x = x ^ 2 - 2) :
    g (f (-1)) = 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg g, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (f - g), sq_nonneg (f + g), mul_self_nonneg (f - g)]
    | simp_all [*]
    | decide