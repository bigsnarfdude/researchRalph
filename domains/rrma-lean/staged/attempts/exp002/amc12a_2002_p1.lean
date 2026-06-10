import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p1 (f : ℂ → ℂ) (h₀ : ∀ x, f x = (2 * x + 3) * (x - 4) + (2 * x + 3) * (x - 6))
  (h₁ : Fintype (f ⁻¹' {0})) : (∑ y ∈ (f ⁻¹' {0}).toFinset, y) = 7 / 2 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (f - h₀), sq_nonneg (f + h₀), mul_self_nonneg (f - h₀)]
    | simp_all [*]
    | decide