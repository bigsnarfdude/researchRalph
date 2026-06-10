import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_132 (x : ℝ) (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x + 2) (h₁ : ∀ x, g x = x ^ 2)
  (h₂ : f (g x) = g (f x)) : x = -1 / 2 := by
  first
    | simp only [h₀, h₁]; field_simp; ring
    | simp only [h₀, h₁]; field_simp; norm_num
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide