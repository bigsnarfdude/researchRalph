import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by
  first
  | solve | ring
  | solve | norm_num
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
  | solve | push_cast; ring
  | solve | push_cast; norm_num