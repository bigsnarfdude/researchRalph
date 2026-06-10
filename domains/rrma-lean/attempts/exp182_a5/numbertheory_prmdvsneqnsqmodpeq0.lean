import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_prmdvsneqnsqmodpeq0 (n : ℤ) (p : ℕ) (h₀ : Nat.Prime p) :
  ↑p ∣ n ↔ n ^ 2 % p = 0 := by
  have hp : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp h₀
  constructor
  · intro h
    have : (↑p : ℤ) ∣ n ^ 2 := dvd_pow h (by norm_num : 2 ≠ 0)
    exact Int.emod_eq_zero_of_dvd this
  · intro h
    have h1 : (↑p : ℤ) ∣ n ^ 2 := Int.dvd_of_emod_eq_zero h
    exact hp.dvd_of_dvd_pow h1
