import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide