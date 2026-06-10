import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_493 (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 - 4 * Real.sqrt x + 1) :
    f (f 4) = 70 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg h₀, sq_nonneg (f - h₀), sq_nonneg (f + h₀), mul_self_nonneg (f - h₀)]
    | simp_all [*]
    | decide