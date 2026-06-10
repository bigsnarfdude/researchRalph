import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by
  first
  | solve | norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num