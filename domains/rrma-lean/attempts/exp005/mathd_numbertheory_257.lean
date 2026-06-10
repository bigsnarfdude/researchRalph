import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_257 (x : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 100)
    (h₁ : 77 ∣ (∑ k ∈ Finset.range 101, k) - x) : x = 45 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all