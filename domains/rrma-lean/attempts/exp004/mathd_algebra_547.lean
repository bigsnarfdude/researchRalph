import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_547 (x y : ℝ) (h₀ : x = 5) (h₁ : y = 2) : Real.sqrt (x ^ 3 - 2 ^ y) = 11 := by
  first
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | simp only [h₀, h₁]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide