import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_prmdvsneqnsqmodpeq0 (n : ℤ) (p : ℕ) (h₀ : Nat.Prime p) :
  ↑p ∣ n ↔ n ^ 2 % p = 0 := by
  first
    | omega
    | constructor <;> intro <;> omega
    | constructor <;> intro <;> simp_all
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide