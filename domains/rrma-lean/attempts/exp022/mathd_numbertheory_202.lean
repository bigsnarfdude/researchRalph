import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by
  first
  | solve | norm_num
  | solve | native_decide
  | solve | decide
  | solve | simp; norm_num
  | solve | simp; native_decide
  | solve | ring
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith