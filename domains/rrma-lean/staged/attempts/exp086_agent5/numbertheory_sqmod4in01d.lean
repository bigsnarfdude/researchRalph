import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod4in01d (a : ℤ) : a ^ 2 % 4 = 0 ∨ a ^ 2 % 4 = 1 := by
  have h : a % 4 = 0 ∨ a % 4 = 1 ∨ a % 4 = 2 ∨ a % 4 = 3 := by omega
  rcases h with h | h | h | h
  all_goals {
    rw [show a ^ 2 = a * a from by ring]
    rw [Int.mul_emod, h]
    simp
  }
