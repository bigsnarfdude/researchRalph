import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_4bap1lt4bsqpap1sq (a b : ℝ) : 4 * b * (a + 1) ≤ 4 * b ^ 2 + (a + 1) ^ 2 := by
  first
  | solve | norm_num
  | solve | linarith
  | solve | nlinarith
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num