import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_110 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ b ≤ a) (h₁ : (a + b) % 10 = 2)
  (h₂ : (2 * a + b) % 10 = 1) : (a - b) % 10 = 6 := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide