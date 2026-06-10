import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide