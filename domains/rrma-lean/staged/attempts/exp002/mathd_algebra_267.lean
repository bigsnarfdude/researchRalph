import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_267 (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -2)
  (h₂ : (x + 1) / (x - 1) = (x - 2) / (x + 2)) : x = 0 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all [*]
    | decide