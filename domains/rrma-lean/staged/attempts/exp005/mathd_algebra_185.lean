import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_185 (s : Finset ℤ) (f : ℤ → ℤ) (h₀ : ∀ x, f x = abs (x + 4))
  (h₁ : ∀ x, x ∈ s ↔ f x < 9) : s.card = 17 := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₀, h₁] at *; nlinarith
    | simp only [h₀, h₁] at *; linarith
    | simp only [h₀, h₁] at *; omega
    | simp only [h₀, h₁] at *; norm_num
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | norm_num
    | ring