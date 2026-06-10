import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_2erprsqpesqeqnrpnesq (e r : ℂ) :
  2 * (e * r) + (e ^ 2 + r ^ 2) = (-r + -e) ^ 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg e, sq_nonneg r, sq_nonneg (e - r), sq_nonneg (e + r), mul_self_nonneg (e - r)]
    | simp_all [*]
    | decide