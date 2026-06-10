import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_28 (c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 + 5 * x + c)
  (h₁ : ∃ x, f x ≤ 0) : c ≤ 25 / 8 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg c, sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (c - f), sq_nonneg (c + f), mul_self_nonneg (c - f)]
    | simp_all [*]
    | decide