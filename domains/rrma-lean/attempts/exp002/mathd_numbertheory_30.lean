import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_30 :
  (33818 ^ 2 + 33819 ^ 2 + 33820 ^ 2 + 33821 ^ 2 + 33822 ^ 2) % 17 = 0 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide