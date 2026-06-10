import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_214 (a : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = a * (x - 2) ^ 2 + 3) (h₁ : f 4 = 4) :
  f 6 = 7 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - f), sq_nonneg (a + f), mul_self_nonneg (a - f)]
    | simp_all [*]
    | decide