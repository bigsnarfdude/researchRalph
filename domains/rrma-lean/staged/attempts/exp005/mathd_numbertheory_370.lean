import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_370 (n : ℕ) (h₀ : n % 7 = 3) : (2 * n + 1) % 7 = 0 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide