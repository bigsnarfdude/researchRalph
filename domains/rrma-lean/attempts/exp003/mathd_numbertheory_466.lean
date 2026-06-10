import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_466 : (∑ k ∈ Finset.range 11, k) % 9 = 1 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | simp [Finset.sum_range_succ]
  | solve | simp [Finset.sum_range_succ]; ring
  | solve | simp [Finset.sum_range_succ]; norm_num
  | solve | native_decide
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