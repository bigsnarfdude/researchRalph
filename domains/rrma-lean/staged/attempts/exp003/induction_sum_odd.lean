import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_sum_odd (n : ℕ) : (∑ k ∈ Finset.range n, (2 * k + 1)) = n ^ 2 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | omega
  | solve | simp [Finset.sum_range_succ]
  | solve | simp [Finset.sum_range_succ]; ring
  | solve | simp [Finset.sum_range_succ]; norm_num
  | solve | native_decide
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