import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_247 (t s : ℝ) (n : ℤ) (h₀ : t = 2 * s - s ^ 2) (h₁ : s = n ^ 2 - 2 ^ n + 1)
  (_ : n = 3) : t = 0 := by
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