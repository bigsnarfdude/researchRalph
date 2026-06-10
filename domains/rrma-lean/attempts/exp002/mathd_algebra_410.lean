import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_410 (x y : ℝ) (h₀ : y = x ^ 2 - 6 * x + 13) : 4 ≤ y := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg h₀, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide