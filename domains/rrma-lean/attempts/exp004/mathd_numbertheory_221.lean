import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_221 (S : Finset ℕ)
  (h₀ : ∀ x : ℕ, x ∈ S ↔ 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) : S.card = 11 := by
  first
    | omega
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide