import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_221 (S : Finset ℕ)
  (h₀ : ∀ x : ℕ, x ∈ S ↔ 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) : S.card = 11 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg h₀]
    | simp_all
    | decide)