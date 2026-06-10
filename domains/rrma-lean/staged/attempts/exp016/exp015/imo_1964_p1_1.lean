import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1964_p1_1 (n : ℕ) (h₀ : 7 ∣ 2 ^ n - 1) : 3 ∣ n := by
  first
  | solve | omega
  | solve | norm_num
  | solve | simp; omega
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀]; omega
  | solve | simp [h₀]; omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
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