import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_480 (f : ℝ → ℝ) (h₀ : ∀ x < 0, f x = -x ^ 2 - 1)
  (h₁ : ∀ x, 0 ≤ x ∧ x < 4 → f x = 2) (h₂ : ∀ x ≥ 4, f x = Real.sqrt x) : f π = 2 := by
  first
    | simp only [h₀, h₁, h₂] at *; nlinarith
    | simp only [h₀, h₁, h₂] at *; linarith
    | simp only [h₀, h₁, h₂] at *; omega
    | simp only [h₀, h₁, h₂] at *; norm_num
    | simp only [h₀, h₁, h₂]; ring
    | simp only [h₀, h₁, h₂]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide