import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by
  first
  | solve | ring
  | solve | ring_nf; norm_num [Complex.I_sq]
  | solve | ring_nf; linear_combination 49 * Complex.I_sq
  | solve | simp [Complex.ext_iff, Complex.I_sq]; constructor <;> ring
  | solve | omega
  | solve | norm_num
  | solve | linarith
  | solve | nlinarith
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