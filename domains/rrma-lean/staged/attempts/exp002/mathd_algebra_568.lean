import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_568 (a : ℝ) :
    (a - 1) * (a + 1) * (a + 2) - (a - 2) * (a + 1) = a ^ 3 + a ^ 2 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a]
    | simp_all [*]
    | decide