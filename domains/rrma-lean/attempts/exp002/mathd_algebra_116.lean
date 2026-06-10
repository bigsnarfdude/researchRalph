import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg k, sq_nonneg x, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (k - x), sq_nonneg (k + x), mul_self_nonneg (k - x)]
    | simp_all [*]
    | decide