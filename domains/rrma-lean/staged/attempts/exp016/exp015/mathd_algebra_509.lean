import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_509 :
  Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = 13 / 6 := by
  first
  | solve | norm_num
  | solve | native_decide
  | solve | decide
  | solve | simp; norm_num
  | solve | simp; native_decide
  | solve | ring
  | solve | omega
  | solve | norm_num [Real.sqrt_lt', Real.lt_sqrt]
  | solve | simp [Real.sqrt_eq_iff_sq_eq]; norm_num
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