import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_509 :
  Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = 13 / 6 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
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