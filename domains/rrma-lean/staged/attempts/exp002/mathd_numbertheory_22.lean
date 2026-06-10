import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_22 (b : ℕ) (h₀ : b < 10)
  (h₁ : Nat.sqrt (10 * b + 6) * Nat.sqrt (10 * b + 6) = 10 * b + 6) : b = 3 ∨ b = 1 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (b - h₀), sq_nonneg (b + h₀), mul_self_nonneg (b - h₀)]
    | simp_all [*]
    | decide