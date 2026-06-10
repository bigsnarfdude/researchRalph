import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  first
    | simp only [h₂] at *; nlinarith
    | simp only [h₂] at *; linarith
    | simp only [h₂] at *; omega
    | simp only [h₂] at *; norm_num
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | simp only [h₂] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all