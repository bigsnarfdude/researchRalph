import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_prmdvsneqnsqmodpeq0 (n : ℤ) (p : ℕ) (h₀ : Nat.Prime p) :
  ↑p ∣ n ↔ n ^ 2 % p = 0 := by
  constructor <;> (first
    | intro h; first | omega | linarith | simp_all [*] | exact h
    | intro h; omega
    | intro h; linarith
    | intro; simp_all [*]
    | intro; norm_num)