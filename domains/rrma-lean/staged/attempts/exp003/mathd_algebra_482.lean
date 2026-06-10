import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num