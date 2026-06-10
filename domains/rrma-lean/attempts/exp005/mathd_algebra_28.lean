import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_28 (c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 + 5 * x + c)
  (h₁ : ∃ x, f x ≤ 0) : c ≤ 25 / 8 := by
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