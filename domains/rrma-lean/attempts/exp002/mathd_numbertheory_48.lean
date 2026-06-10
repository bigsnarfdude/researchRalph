import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_48 (b : ℕ) (h₀ : 0 < b) (h₁ : 3 * b ^ 2 + 2 * b + 1 = 57) : b = 4 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg b, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (b - h₀), sq_nonneg (b + h₀), mul_self_nonneg (b - h₀)]
    | simp_all [*]
    | decide