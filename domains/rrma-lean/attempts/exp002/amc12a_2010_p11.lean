import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p11 (x b : ℝ) (h₀ : 0 < b) (h₁ : (7 : ℝ) ^ (x + 7) = 8 ^ x)
  (h₂ : x = Real.logb b (7 ^ 7)) : b = 8 / 7 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - b), sq_nonneg (x + b), mul_self_nonneg (x - b)]
    | simp_all [*]
    | decide