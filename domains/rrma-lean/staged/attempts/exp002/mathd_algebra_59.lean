import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_59 (b : ℝ) (h₀ : (4 : ℝ) ^ b + 2 ^ 3 = 12) : b = 1 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg b, sq_nonneg h₀, sq_nonneg (b - h₀), sq_nonneg (b + h₀), mul_self_nonneg (b - h₀)]
    | simp_all [*]
    | decide