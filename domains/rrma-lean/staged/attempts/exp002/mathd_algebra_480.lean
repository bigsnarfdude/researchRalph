import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_480 (f : ℝ → ℝ) (h₀ : ∀ x < 0, f x = -x ^ 2 - 1)
  (h₁ : ∀ x, 0 ≤ x ∧ x < 4 → f x = 2) (h₂ : ∀ x ≥ 4, f x = Real.sqrt x) : f π = 2 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (f - h₀), sq_nonneg (f + h₀), mul_self_nonneg (f - h₀)]
    | simp_all
    | decide)