import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg m, sq_nonneg n, sq_nonneg k, sq_nonneg f, sq_nonneg (m - n), sq_nonneg (m + n), mul_self_nonneg (m - n)]
    | simp_all [*]
    | decide