import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  first
    | omega
    | norm_num
    | simp only [h₂] at *; nlinarith
    | simp only [h₂] at *; linarith
    | simp only [h₂] at *; omega
    | simp only [h₂] at *; norm_num
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | ring
    | linarith
    | simp_all
    | decide