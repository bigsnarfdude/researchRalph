import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_257 (x : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 100)
    (h₁ : 77 ∣ (∑ k ∈ Finset.range 101, k) - x) : x = 45 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide