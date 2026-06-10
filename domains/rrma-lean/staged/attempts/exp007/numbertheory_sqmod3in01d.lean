import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod3in01d (a : ℤ) : a ^ 2 % 3 = 0 ∨ a ^ 2 % 3 = 1 := by
  first
  | solve | norm_num
  | solve | left; omega
  | solve | left; norm_num
  | solve | right; omega
  | solve | right; norm_num
  | solve | native_decide
  | solve | decide
  | solve | omega
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num