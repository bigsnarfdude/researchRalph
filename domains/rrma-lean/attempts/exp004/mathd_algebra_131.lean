import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_131 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 - 7 * x + 2)
  (h₁ : f a = 0) (h₂ : f b = 0) (h₃ : a ≠ b) : 1 / (a - 1) + 1 / (b - 1) = -1 := by
  first
    | simp only [h₀]; field_simp; ring
    | simp only [h₀]; field_simp; norm_num
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide