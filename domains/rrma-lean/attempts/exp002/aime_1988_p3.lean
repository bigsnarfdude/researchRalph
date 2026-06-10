import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1988_p3 (x : ℝ) (h₀ : 0 < x)
  (h₁ : Real.logb 2 (Real.logb 8 x) = Real.logb 8 (Real.logb 2 x)) : Real.logb 2 x ^ 2 = 27 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all [*]
    | decide
