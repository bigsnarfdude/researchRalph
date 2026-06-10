import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_709 (n : ℕ) (h₀ : 0 < n) (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30) : Finset.card (Nat.divisors (6 * n)) = 35 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide