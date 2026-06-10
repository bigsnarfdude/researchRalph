import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_36azm9asqle36zsq (z a : ℝ) : 36 * (a * z) - 9 * a ^ 2 ≤ 36 * z ^ 2 := by
  first
    | norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide