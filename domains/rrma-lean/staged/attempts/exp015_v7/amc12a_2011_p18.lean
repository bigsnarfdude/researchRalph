import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2011_p18 (x y : ℝ) (h₀ : abs (x + y) + abs (x - y) = 2) :
  x ^ 2 - 6 * x + y ^ 2 ≤ 8 := by
  try   rw [abs_of_nonneg (by linarith [h₀])]; linarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_nonneg (by linarith)]; linarith
  try   rw [abs_of_nonneg (by linarith [h₀])]; nlinarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_nonneg (by linarith)]; nlinarith
  try   rw [abs_of_nonneg (by linarith [h₀])]; omega [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_nonneg (by linarith)]; omega
  try   rw [abs_of_nonneg (by linarith [h₀])]; norm_num [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_nonneg (by linarith)]; norm_num
  try   rw [abs_of_neg (by linarith [h₀])]; linarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_neg (by linarith)]; linarith
  try   rw [abs_of_neg (by linarith [h₀])]; nlinarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_neg (by linarith)]; nlinarith
  try   rw [abs_of_neg (by linarith [h₀])]; omega [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_neg (by linarith)]; omega
  try   rw [abs_of_neg (by linarith [h₀])]; norm_num [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_neg (by linarith)]; norm_num
  try   rw [abs_of_pos (by linarith [h₀])]; linarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_pos (by linarith)]; linarith
  try   rw [abs_of_pos (by linarith [h₀])]; nlinarith [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_pos (by linarith)]; nlinarith
  try   rw [abs_of_pos (by linarith [h₀])]; omega [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_pos (by linarith)]; omega
  try   rw [abs_of_pos (by linarith [h₀])]; norm_num [h₀]
  try   simp only [abs_le]; constructor <;> linarith [h₀]
  try   rw [abs_of_pos (by linarith)]; norm_num
  first
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀]; ring
  | solve | simp [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | simp [h₀]; omega
  | solve | simp only [h₀]; linarith
  | solve | simp [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | simp [h₀]; nlinarith
  | solve | linear_combination h₀
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
  | solve | field_simp; norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp