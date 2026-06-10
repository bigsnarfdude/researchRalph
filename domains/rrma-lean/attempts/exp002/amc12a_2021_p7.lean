import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2021_p7 (x y : ℝ) : 1 ≤ (x * y - 1) ^ 2 + (x + y) ^ 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide