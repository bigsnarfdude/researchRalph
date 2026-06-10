import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_48 (q e : ℂ) (h₀ : q = 9 - 4 * Complex.I) (h₁ : e = -3 - 4 * Complex.I) :
  q - e = 12 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg q, sq_nonneg e, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (q - e), sq_nonneg (q + e), mul_self_nonneg (q - e)]
    | simp_all [*]
    | decide