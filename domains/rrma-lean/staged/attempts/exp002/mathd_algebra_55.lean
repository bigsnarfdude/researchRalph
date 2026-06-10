import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_55 (q p : ℝ) (h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14)
  (h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21) : q / p = 2 / 3 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg q, sq_nonneg p, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (q - p), sq_nonneg (q + p), mul_self_nonneg (q - p)]
    | simp_all [*]
    | decide