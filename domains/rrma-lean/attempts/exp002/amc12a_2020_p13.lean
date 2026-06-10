import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p13 (a b c : ℕ) (n : NNReal) (h₀ : n ≠ 1) (h₁ : 1 < a ∧ 1 < b ∧ 1 < c)
  (h₂ : (n * (n * n ^ (1 / c)) ^ (1 / b)) ^ (1 / a) = (n ^ 25) ^ (1 / 36)) : b = 3 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg c, sq_nonneg n, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)