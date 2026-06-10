import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_269 : (2005 ^ 2 + 2005 ^ 0 + 2005 ^ 0 + 2005 ^ 5) % 100 = 52 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | ring
    | linarith
    | simp_all