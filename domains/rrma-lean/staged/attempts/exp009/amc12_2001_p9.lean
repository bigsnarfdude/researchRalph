import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
    f 600 = 5 / 2 := by
  first
  | solve | simp only [h₀] at *; ring
  | solve | simp only [h₀] at *; norm_num
  | solve | simp only [h₀] at *; omega
  | solve | simp only [h₀] at *; linarith
  | solve | simp only [h₀] at *; nlinarith
  | solve | simp only [h₀] at *; field_simp; ring
  | solve | simp only [h₀] at *; field_simp; linarith
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
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