import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_421 (a b c d : ℝ) (h₀ : b = a ^ 2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a ^ 2 + a + 6) (h₂ : d = c ^ 2 + 4 * c + 6) (h₃ : d = 1 / 2 * c ^ 2 + c + 6)
  (h₄ : a < c) : c - a = 6 := by
  first
    | simp only [h₀, h₁, h₂, h₃]; ring
    | simp only [h₀, h₁, h₂, h₃]; norm_num
    | simp only [h₀, h₁, h₂, h₃]; omega
    | simp only [h₀, h₁, h₂, h₃]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide