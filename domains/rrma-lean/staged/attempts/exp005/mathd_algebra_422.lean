import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_422 (x : ℝ) (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 5 * x - 12)
  (h₁ : σ.1 (x + 1) = σ.2 x) : x = 47 / 24 := by
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