import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_30 :
  (33818 ^ 2 + 33819 ^ 2 + 33820 ^ 2 + 33821 ^ 2 + 33822 ^ 2) % 17 = 0 := by
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