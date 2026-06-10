import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1991_p1 (x y : ℕ) (h₀ : 0 < x ∧ 0 < y) (h₁ : x * y + (x + y) = 71)
  (h₂ : x ^ 2 * y + x * y ^ 2 = 880) : x ^ 2 + y ^ 2 = 146 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all
    | decide)