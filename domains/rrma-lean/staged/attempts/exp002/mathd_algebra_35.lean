import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_35 (p q : ℝ → ℝ) (h₀ : ∀ x, p x = 2 - x ^ 2)
    (h₁ : ∀ x : ℝ, x ≠ 0 → q x = 6 / x) : p (q 2) = -7 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg p, sq_nonneg q, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (p - q), sq_nonneg (p + q), mul_self_nonneg (p - q)]
    | simp_all [*]
    | decide