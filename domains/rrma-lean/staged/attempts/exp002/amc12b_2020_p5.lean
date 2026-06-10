import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12b_2020_p5 (a b : ℕ) (h₀ : (5 : ℚ) / 8 * b = 2 / 3 * a + 7)
  (h₁ : (b : ℚ) - 5 / 8 * b = a - 2 / 3 * a + 7) : a = 42 := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide