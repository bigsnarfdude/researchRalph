import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_2erprsqpesqeqnrpnesq (e r : ℂ) :
  2 * (e * r) + (e ^ 2 + r ^ 2) = (-r + -e) ^ 2 := by
  first
    | ring
    | norm_num
    | simp_all