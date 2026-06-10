import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_451 (σ : Equiv ℝ ℝ) (h₀ : σ.2 (-15) = 0) (h₁ : σ.2 0 = 3) (h₂ : σ.2 3 = 9)
    (h₃ : σ.2 9 = 20) : σ.1 (σ.1 9) = 0 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg σ, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (σ - h₀), sq_nonneg (σ + h₀), mul_self_nonneg (σ - h₀)]
    | simp_all [*]
    | decide