import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p5 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x - p = 2 - 2 * p := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg p, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - p), sq_nonneg (x + p), mul_self_nonneg (x - p)]
    | simp_all [*]
    | decide