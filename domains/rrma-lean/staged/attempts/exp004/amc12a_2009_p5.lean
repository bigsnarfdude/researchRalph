import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p5 (x : ℝ) (h₀ : x ^ 3 - (x + 1) * (x - 1) * x = 5) : x ^ 3 = 125 := by
  first
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide