import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aimeII_2001_p3 (x : ℕ → ℤ) (h₀ : x 1 = 211) (h₂ : x 2 = 375) (h₃ : x 3 = 420)
  (h₄ : x 4 = 523) (h₆ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4)) :
  x 531 + x 753 + x 975 = 898 := by
  first
  | solve | simp only [h₆] at *; ring
  | solve | simp only [h₆] at *; norm_num
  | solve | simp only [h₆] at *; omega
  | solve | simp only [h₆] at *; linarith
  | solve | simp only [h₆] at *; nlinarith
  | solve | linarith [h₀, h₂, h₃, h₄, h₆]
  | solve | nlinarith [h₀, h₂, h₃, h₄, h₆]
  | solve | linarith
  | solve | nlinarith
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num