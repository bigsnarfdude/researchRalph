import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_36azm9asqle36zsq (z a : ℝ) : 36 * (a * z) - 9 * a ^ 2 ≤ 36 * z ^ 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg z, sq_nonneg a, sq_nonneg (z - a), sq_nonneg (z + a), mul_self_nonneg (z - a)]
    | simp_all [*]
    | decide