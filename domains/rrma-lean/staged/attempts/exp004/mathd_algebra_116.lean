import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by
  first
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀]; linarith
    | field_simp; linarith [h₀, h₁]
    | field_simp; ring
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all