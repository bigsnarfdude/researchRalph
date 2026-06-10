import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_43 : IsGreatest { n : ℕ | 15 ^ n ∣ 942! } 233 := by
  first
    | norm_num
    | native_decide
    | decide
    | ring
    | omega
    | linarith
    | simp_all