import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod3in01d (a : ℤ) : a ^ 2 % 3 = 0 ∨ a ^ 2 % 3 = 1 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a]
    | simp_all [*]
    | decide