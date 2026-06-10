import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_51 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a + b = 35) (h₂ : a = 2 / 5 * b) :
    b - a = 15 := by
  first
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | simp only [h₂]; omega
    | simp only [h₂]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide