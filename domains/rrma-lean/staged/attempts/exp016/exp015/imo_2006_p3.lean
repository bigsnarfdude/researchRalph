import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_2006_p3 (a b c : ℝ) :
  a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2) ≤
  9 * Real.sqrt 2 / 32 * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2 := by
  first
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | field_simp; nlinarith
  | solve | field_simp; linarith
  | solve | norm_num
  | solve | ring
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
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