import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_109 (a b : ℝ) (h₀ : 3 * a + 2 * b = 12) (h₁ : a = 4) : b = 0 := by
  first
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | simp only [h₁]; omega
    | simp only [h₁]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide