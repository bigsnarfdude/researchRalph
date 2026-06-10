import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_149 (f : ℝ → ℝ) (h₀ : ∀ x < -5, f x = x ^ 2 + 5)
  (h₁ : ∀ x ≥ -5, f x = 3 * x - 8) (h₂ : Fintype (f ⁻¹' {10})) :
  (∑ k ∈ (f ⁻¹' {10}).toFinset, k) = 6 := by
  first
  | solve | simp only [h₀, h₁] at *; ring
  | solve | simp only [h₀, h₁] at *; norm_num
  | solve | simp only [h₀, h₁] at *; omega
  | solve | simp only [h₀, h₁] at *; linarith
  | solve | simp only [h₀, h₁] at *; nlinarith
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; omega
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | nlinarith [sq_nonneg _, h₀, h₁, h₂]
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