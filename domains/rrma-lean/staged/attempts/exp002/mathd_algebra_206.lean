import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_206 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 + a * x + b) (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0) (h₃ : f b = 0) : a + b = -1 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg f, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide