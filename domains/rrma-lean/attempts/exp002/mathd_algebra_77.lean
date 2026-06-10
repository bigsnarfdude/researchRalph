import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg f, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)