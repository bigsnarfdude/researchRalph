import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p12 (f : ℝ → ℝ) (k : ℝ) (a b : ℕ) (h₀ : ∀ x, f x = x ^ 2 - 63 * x + k)
  (h₁ : f a = 0 ∧ f b = 0) (h₂ : a ≠ b) (h₃ : Nat.Prime a ∧ Nat.Prime b) : k = 122 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg k, sq_nonneg a, sq_nonneg b, sq_nonneg (f - k), sq_nonneg (f + k), mul_self_nonneg (f - k)]
    | simp_all
    | decide)