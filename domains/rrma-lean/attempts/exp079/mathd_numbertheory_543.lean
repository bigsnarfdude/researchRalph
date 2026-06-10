import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_543 : (∑ k ∈ Nat.divisors (30 ^ 4), 1) - 2 = 123 := by
  first
  | solve | native_decide
  | solve | norm_num
  | solve | decide
  | solve | simp; norm_num
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
  | solve | push_cast; ring
  | solve | push_cast; norm_num