import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1984_p5 (a b : ℝ) (h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5)
  (h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7) : a * b = 512 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide