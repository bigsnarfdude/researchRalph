import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_2erprsqpesqeqnrpnesq (e r : ℂ) :
  2 * (e * r) + (e ^ 2 + r ^ 2) = (-r + -e) ^ 2 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | push_cast; ring
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; norm_num