import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_55 (q p : ℝ) (h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14)
  (h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21) : q / p = 2 / 3 := by
  first
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | simp only [h₀, h₁]; linarith
    | field_simp; linarith [h₀, h₁]
    | field_simp; ring
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all