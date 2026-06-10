import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_binomnegdiscrineq_10alt28asqp1 (a : ℝ) : 10 * a ≤ 28 * a ^ 2 + 1 := by
  first
    | norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide