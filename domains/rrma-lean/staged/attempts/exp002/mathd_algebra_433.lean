import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_433 (f : ℝ → ℝ) (h₀ : ∀ x, f x = 3 * Real.sqrt (2 * x - 7) - 8) : f 8 = 19 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg h₀, sq_nonneg (f - h₀), sq_nonneg (f + h₀), mul_self_nonneg (f - h₀)]
    | simp_all [*]
    | decide