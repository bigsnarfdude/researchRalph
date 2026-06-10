import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
    f 600 = 5 / 2 := by
  first
    | simp only [h₀] at *; nlinarith
    | simp only [h₀] at *; linarith
    | simp only [h₀] at *; omega
    | simp only [h₀] at *; norm_num
    | simp only [h₀] at *; field_simp; ring
    | simp only [h₀] at *; field_simp; linarith
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | omega
    | linarith