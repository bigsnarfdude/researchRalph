import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2011_p18 (x y : ℝ) (h₀ : abs (x + y) + abs (x - y) = 2) :
  x ^ 2 - 6 * x + y ^ 2 ≤ 8 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg h₀, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide