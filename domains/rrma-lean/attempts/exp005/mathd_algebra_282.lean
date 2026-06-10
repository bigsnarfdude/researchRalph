import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_282 (f : ℝ → ℝ) (h₀ : ∀ x : ℝ, ¬ (Irrational x) → f x = abs (Int.floor x))
  (h₁ : ∀ x, Irrational x → f x = (Int.ceil x) ^ 2) :
  f (8 ^ (1 / 3)) + f (-Real.pi) + f (Real.sqrt 50) + f (9 / 2) = 79 := by
  first
    | simp only [h₀, h₁] at *; nlinarith
    | simp only [h₀, h₁] at *; linarith
    | simp only [h₀, h₁] at *; omega
    | simp only [h₀, h₁] at *; norm_num
    | simp only [h₀, h₁] at *; field_simp; ring
    | simp only [h₀, h₁] at *; field_simp; linarith
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | norm_num
    | ring
    | omega