import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod3in01d (a : ℤ) : a ^ 2 % 3 = 0 ∨ a ^ 2 % 3 = 1 := by
  have h1 : a ^ 2 % 3 = (a % 3 * (a % 3)) % 3 := by
    rw [show a ^ 2 = a * a from by ring, Int.mul_emod]
  have h2 : a % 3 = 0 ∨ a % 3 = 1 ∨ a % 3 = 2 := by omega
  rcases h2 with h | h | h <;> simp [h1, h] <;> omega
