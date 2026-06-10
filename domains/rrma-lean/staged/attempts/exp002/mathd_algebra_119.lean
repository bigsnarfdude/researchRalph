import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_119 (d e : ℝ) (h₀ : 2 * d = 17 * e - 8) (h₁ : 2 * e = d - 9) : e = 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg d, sq_nonneg e, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (d - e), sq_nonneg (d + e), mul_self_nonneg (d - e)]
    | simp_all [*]
    | decide