import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_2dvd4expn (n : ℕ) (h₀ : n ≠ 0) : 2 ∣ 4 ^ n := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide