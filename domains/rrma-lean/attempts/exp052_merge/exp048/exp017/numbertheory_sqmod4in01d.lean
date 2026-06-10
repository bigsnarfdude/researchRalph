import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sqmod4in01d (a : ℤ) : a ^ 2 % 4 = 0 ∨ a ^ 2 % 4 = 1 := by
  have h : a % 2 = 0 ∨ a % 2 = 1 := by omega
  rcases h with h | h
  · -- a is even: a = 2k
    left
    have ⟨k, hk⟩ := (Int.dvd_iff_emod_eq_zero.mpr h)
    rw [show a = 2 * (a / 2) from by omega] at *
    ring_nf
    omega
  · -- a is odd: a = 2k + 1
    right
    rw [show a = 2 * (a / 2) + 1 from by omega]
    ring_nf
    omega
