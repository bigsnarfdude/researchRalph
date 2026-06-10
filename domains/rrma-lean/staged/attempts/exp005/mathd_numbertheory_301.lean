import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_301 (j : ℕ) (h₀ : 0 < j) : 3 * (7 * ↑j + 3) % 7 = 2 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide