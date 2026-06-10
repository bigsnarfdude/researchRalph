import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_234 (d : ℝ) (h₀ : 27 / 125 * d = 9 / 25) : 3 / 5 * d ^ 3 = 25 / 9 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg d, sq_nonneg h₀, sq_nonneg (d - h₀), sq_nonneg (d + h₀), mul_self_nonneg (d - h₀)]
    | simp_all [*]
    | decide