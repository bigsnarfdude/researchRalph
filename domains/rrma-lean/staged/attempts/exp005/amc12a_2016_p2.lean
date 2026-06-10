import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2016_p2 (x : ℝ) (h₀ : (10 : ℝ) ^ x * 100 ^ (2 * x) = 1000 ^ 5) : x = 3 := by
  first
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide