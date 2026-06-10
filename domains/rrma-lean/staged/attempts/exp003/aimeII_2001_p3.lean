import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aimeII_2001_p3 (x : ℕ → ℤ) (h₀ : x 1 = 211) (h₂ : x 2 = 375) (h₃ : x 3 = 420)
  (h₄ : x 4 = 523) (h₆ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4)) :
  x 531 + x 753 + x 975 = 898 := by
  first
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; ring
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; norm_num
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; linarith
  | solve | simp only [h₀, h₂, h₃, h₄, h₆]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₂, h₃, h₄, h₆]
  | solve | nlinarith [h₀, h₂, h₃, h₄, h₆]
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num