import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p15 (f : ℂ → ℂ) (h₀ : ∀ x, f (x / 3) = x ^ 2 + x + 1)
  (h₁ : Fintype (f ⁻¹' {7})) : (∑ y ∈ (f ⁻¹' {7}).toFinset, y / 3) = -1 / 9 := by
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