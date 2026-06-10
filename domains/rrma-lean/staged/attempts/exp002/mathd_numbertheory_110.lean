import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_110 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ b ≤ a) (h₁ : (a + b) % 10 = 2)
  (h₂ : (2 * a + b) % 10 = 1) : (a - b) % 10 = 6 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)