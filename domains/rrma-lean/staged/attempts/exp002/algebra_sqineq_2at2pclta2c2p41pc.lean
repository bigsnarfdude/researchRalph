import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_2at2pclta2c2p41pc (a c : ℝ) :
  2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg c, sq_nonneg (a - c), sq_nonneg (a + c), mul_self_nonneg (a - c)]
    | simp_all [*]
    | decide