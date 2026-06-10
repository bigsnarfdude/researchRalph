import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod3in01d (a : ℤ) : a ^ 2 % 3 = 0 ∨ a ^ 2 % 3 = 1 := by
  have h : a % 3 = 0 ∨ a % 3 = 1 ∨ a % 3 = 2 := by omega
  rcases h with h | h | h
  all_goals {
    rw [show a ^ 2 = a * a from by ring]
    rw [Int.mul_emod, h]
    simp
  }
