import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_132 (x : ℝ) (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x + 2) (h₁ : ∀ x, g x = x ^ 2)
  (h₂ : f (g x) = g (f x)) : x = -1 / 2 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg f, sq_nonneg g, sq_nonneg h₀, sq_nonneg (x - f), sq_nonneg (x + f), mul_self_nonneg (x - f)]
    | simp_all [*]
    | decide