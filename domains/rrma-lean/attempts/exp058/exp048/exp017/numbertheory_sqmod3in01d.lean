import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod3in01d (a : ℤ) : a ^ 2 % 3 = 0 ∨ a ^ 2 % 3 = 1 := by
  have h : a % 3 = 0 ∨ a % 3 = 1 ∨ a % 3 = 2 := by omega
  rcases h with h | h | h
  · left
    rw [show a = 3 * (a / 3) from by omega]
    ring_nf; omega
  · right
    rw [show a = 3 * (a / 3) + 1 from by omega]
    ring_nf; omega
  · right
    rw [show a = 3 * (a / 3) + 2 from by omega]
    ring_nf; omega
