import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1990_p2 :
  (52 + 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) - (52 - 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) = 828 := by
  try   rw [Real.sqrt_sq] <;> [linarith, linarith]
  try   simp [Real.sqrt_sq]; linarith
  try   rw [Real.sqrt_sq] <;> [nlinarith, linarith]
  try   simp [Real.sqrt_sq]; nlinarith
  try   rw [Real.sqrt_sq] <;> [ring, linarith]
  try   simp [Real.sqrt_sq]; ring
  try   rw [Real.sqrt_sq] <;> [norm_num, linarith]
  try   simp [Real.sqrt_sq]; norm_num
  try   rw [Real.sq_sqrt] <;> [linarith, linarith]
  try   simp [Real.sq_sqrt]; linarith
  try   rw [Real.sq_sqrt] <;> [nlinarith, linarith]
  try   simp [Real.sq_sqrt]; nlinarith
  try   rw [Real.sq_sqrt] <;> [ring, linarith]
  try   simp [Real.sq_sqrt]; ring
  try   rw [Real.sq_sqrt] <;> [norm_num, linarith]
  try   simp [Real.sq_sqrt]; norm_num
  try   rw [Real.sqrt_eq_iff_sq_eq] <;> [linarith, linarith]
  try   simp [Real.sqrt_eq_iff_sq_eq]; linarith
  try   rw [Real.sqrt_eq_iff_sq_eq] <;> [nlinarith, linarith]
  try   simp [Real.sqrt_eq_iff_sq_eq]; nlinarith
  first
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | ring_nf; omega
  | solve | ring_nf; norm_num
  | solve | ring_nf; ring
  | solve | ring_nf; linarith
  | solve | ring_nf; nlinarith
  | solve | ring_nf; simp
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; ring
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; simp
  | solve | push_cast; omega
  | solve | push_cast; norm_num
  | solve | push_cast; ring
  | solve | push_cast; linarith
  | solve | push_cast; nlinarith
  | solve | push_cast; simp
  | solve | norm_cast; omega
  | solve | norm_cast; norm_num
  | solve | norm_cast; ring
  | solve | norm_cast; linarith
  | solve | norm_cast; nlinarith
  | solve | norm_cast; simp
  | solve | field_simp; omega
  | solve | field_simp; simp