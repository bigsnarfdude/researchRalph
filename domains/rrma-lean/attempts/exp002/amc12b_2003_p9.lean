import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12b_2003_p9 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = a * x + b) (h₁ : f 6 - f 2 = 12) :
    f 12 - f 2 = 30 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg f, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide