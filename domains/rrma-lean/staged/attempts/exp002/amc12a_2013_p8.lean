import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2013_p8 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y) : x * y = 2 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide