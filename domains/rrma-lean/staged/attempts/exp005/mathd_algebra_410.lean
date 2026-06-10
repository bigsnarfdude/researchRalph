import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_410 (x y : ℝ) (h₀ : y = x ^ 2 - 6 * x + 13) : 4 ≤ y := by
  first
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀]; omega
    | simp only [h₀]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide