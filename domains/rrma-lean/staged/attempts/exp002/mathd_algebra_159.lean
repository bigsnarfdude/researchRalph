import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_159 (b : ℝ) (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * x ^ 4 - 7 * x ^ 3 + 2 * x ^ 2 - b * x + 1) (h₁ : f 1 = 1) : b = -2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg b, sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (b - f), sq_nonneg (b + f), mul_self_nonneg (b - f)]
    | simp_all [*]
    | decide