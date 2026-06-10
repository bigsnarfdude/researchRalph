import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_422 (x : ℝ) (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 5 * x - 12)
  (h₁ : σ.1 (x + 1) = σ.2 x) : x = 47 / 24 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg σ, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - σ), sq_nonneg (x + σ), mul_self_nonneg (x - σ)]
    | simp_all [*]
    | decide