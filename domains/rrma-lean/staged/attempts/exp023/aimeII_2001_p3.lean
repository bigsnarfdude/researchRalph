import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aimeII_2001_p3 (x : ℕ → ℤ) (h₀ : x 1 = 211) (h₂ : x 2 = 375) (h₃ : x 3 = 420)
  (h₄ : x 4 = 523) (h₆ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4)) :
  x 531 + x 753 + x 975 = 898 := by
  first
  | solve | linarith [h₀, h₂, h₃, h₄, h₆]
  | solve | nlinarith [h₀, h₂, h₃, h₄, h₆]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | nlinarith [sq_nonneg x, h₀, h₂, h₃, h₄, h₆]
  | solve | nlinarith [sq_nonneg (x - 1), h₀, h₂, h₃, h₄, h₆]
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; ring
  | solve | simp [h₀, h₂, h₃, h₄, h₆]; ring
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; norm_num
  | solve | simp [h₀, h₂, h₃, h₄, h₆]; norm_num
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; omega
  | solve | simp [h₀, h₂, h₃, h₄, h₆]; omega
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; linarith
  | solve | simp [h₀, h₂, h₃, h₄, h₆]; linarith
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; nlinarith
  | solve | simp [h₀, h₂, h₃, h₄, h₆]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₂
  | solve | linear_combination h₃
  | solve | linear_combination h₀ + h₂
  | solve | linear_combination h₀ + -h₂
  | solve | linear_combination -h₀ + h₂
  | solve | linear_combination 2 * h₀ + -h₂
  | solve | linear_combination -h₀ + 2 * h₂
  | solve | linear_combination 3 * h₀ + -h₂
  | solve | linear_combination -3 * h₀ + 2 * h₂
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