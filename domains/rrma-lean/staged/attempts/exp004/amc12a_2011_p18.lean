import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2011_p18 (x y : ℝ) (h₀ : abs (x + y) + abs (x - y) = 2) :
  x ^ 2 - 6 * x + y ^ 2 ≤ 8 := by
  first
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide