import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_67 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = 5 * x + 3) (h₁ : ∀ x, g x = x ^ 2 - 2) :
    g (f (-1)) = 2 := by
  first
    | simp only [h₀, h₁] at *; nlinarith
    | simp only [h₀, h₁] at *; linarith
    | simp only [h₀, h₁] at *; omega
    | simp only [h₀, h₁] at *; norm_num
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide