import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_109 (a b : ℝ) (h₀ : 3 * a + 2 * b = 12) (h₁ : a = 4) : b = 0 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all [*]
    | decide