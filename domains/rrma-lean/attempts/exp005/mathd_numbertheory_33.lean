import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_33 (n : ℕ) (h₀ : n < 398) (h₁ : n * 7 % 398 = 1) : n = 57 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide