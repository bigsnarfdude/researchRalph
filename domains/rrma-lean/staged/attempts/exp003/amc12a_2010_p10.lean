import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p10 (p q : ℝ) (a : ℕ → ℝ) (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p) (h₂ : a 2 = 9) (h₃ : a 3 = 3 * p - q) (h₄ : a 4 = 3 * p + q) : a 2010 = 8041 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄]
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