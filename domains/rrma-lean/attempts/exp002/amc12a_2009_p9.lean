import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p9 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 3) = 3 * x ^ 2 + 7 * x + 4)
  (h₁ : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg c, sq_nonneg f, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide