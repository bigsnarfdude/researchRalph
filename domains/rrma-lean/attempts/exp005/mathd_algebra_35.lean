import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_35 (p q : ℝ → ℝ) (h₀ : ∀ x, p x = 2 - x ^ 2)
    (h₁ : ∀ x : ℝ, x ≠ 0 → q x = 6 / x) : p (q 2) = -7 := by
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