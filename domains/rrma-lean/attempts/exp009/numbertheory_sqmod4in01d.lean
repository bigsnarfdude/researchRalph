import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod4in01d (a : ℤ) : a ^ 2 % 4 = 0 ∨ a ^ 2 % 4 = 1 := by
  have h1 : a ^ 2 % 4 = (a % 4 * (a % 4)) % 4 := by
    rw [show a ^ 2 = a * a from by ring, Int.mul_emod]
  have h2 : a % 4 = 0 ∨ a % 4 = 1 ∨ a % 4 = 2 ∨ a % 4 = 3 := by omega
  rcases h2 with h | h | h | h <;> simp [h1, h] <;> omega
