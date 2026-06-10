import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_393 (σ : Equiv ℝ ℝ) (h₀ : ∀ x, σ.1 x = 4 * x ^ 3 + 1) : σ.2 33 = 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg σ, sq_nonneg h₀, sq_nonneg (σ - h₀), sq_nonneg (σ + h₀), mul_self_nonneg (σ - h₀)]
    | simp_all [*]
    | decide