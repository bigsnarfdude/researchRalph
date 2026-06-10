import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12_2001_p2 (a b n : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : n = 10 * a + b)
  (h₃ : n = a * b + a + b) : b = 9 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg n, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)