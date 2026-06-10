import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_257 (x : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 100)
    (h₁ : 77 ∣ (∑ k ∈ Finset.range 101, k) - x) : x = 45 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all
    | decide)