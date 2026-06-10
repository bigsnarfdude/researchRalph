import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_11 (a b : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ 2 * b)
    (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) : (a + 11 * b) / (a - b) = 2 := by
  first
    | field_simp; linarith [h₀, h₁, h₂]
    | field_simp; nlinarith [h₀, h₁, h₂]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide