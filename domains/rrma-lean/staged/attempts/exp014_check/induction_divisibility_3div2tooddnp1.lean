import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  first
  | solve | norm_num
  | solve | omega
  | solve | simp; omega
  | solve | native_decide
  | solve | decide
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
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num