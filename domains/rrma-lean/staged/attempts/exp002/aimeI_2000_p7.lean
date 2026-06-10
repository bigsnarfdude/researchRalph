import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aimeI_2000_p7 (x y z : ℝ) (m : ℚ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x * y * z = 1)
  (h₂ : x + 1 / z = 5) (h₃ : y + 1 / x = 29) (h₄ : z + 1 / y = m) (h₅ : 0 < m) :
  ↑m.den + m.num = 5 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z, sq_nonneg m, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all
    | decide)