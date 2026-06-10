import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_136 (n : ℕ) (h₀ : 123 * n + 17 = 39500) : n = 321 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide