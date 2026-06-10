import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  first
  | solve | norm_num
  | solve | simp only [h₂] at *; ring
  | solve | simp only [h₂] at *; norm_num
  | solve | simp only [h₂] at *; omega
  | solve | simp only [h₂] at *; linarith
  | solve | simp only [h₂] at *; nlinarith
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num