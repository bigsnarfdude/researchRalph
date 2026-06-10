import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem numbertheory_2dvd4expn (n : ℕ) (h₀ : n ≠ 0) : 2 ∣ 4 ^ n := by
  have : 2 ∣ 4 := by norm_num
  exact dvd_trans this (dvd_pow_self 4 h₀)
