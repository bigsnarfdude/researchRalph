import Mathlib
set_option maxHeartbeats 16000000

open Real

theorem imo_1965_p1 (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 2 * π)
  (h₂ : 2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))))
  (h₃ : abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := by
  -- From h₂ and h₃: 2*cos(x) ≤ √2, so cos(x) ≤ √2/2 = cos(π/4)
  have hcos : Real.cos x ≤ Real.sqrt 2 / 2 := by linarith
  constructor
  · -- π/4 ≤ x
    by_contra hlt
    push_neg at hlt
    -- x ∈ [0, π/4), so cos(x) > cos(π/4) = √2/2
    have hx_pos : 0 ≤ x := h₀
    have hx_lt : x < π / 4 := hlt
    have : Real.cos (π / 4) < Real.cos x :=
      Real.cos_lt_cos_of_nonneg_of_le_pi (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos]) (by linarith)
    have : Real.cos (π / 4) = Real.sqrt 2 / 2 := Real.cos_pi_div_four
    linarith
  · -- x ≤ 7π/4
    by_contra hgt
    push_neg at hgt
    -- x ∈ (7π/4, 2π], so cos(x) > cos(7π/4) = cos(2π - π/4) = cos(π/4) = √2/2
    have hx_gt : 7 * π / 4 < x := hgt
    have hx_le : x ≤ 2 * π := h₁
    -- cos(2π - y) = cos(y), so cos(x) = cos(2π-x) where 2π-x ∈ [0, π/4)
    have h2pi_sub : Real.cos x = Real.cos (2 * π - x) := by
      rw [Real.cos_sub, Real.cos_two_pi, Real.sin_two_pi]; ring
    have h_range : 0 ≤ 2 * π - x := by linarith
    have h_range2 : 2 * π - x < π / 4 := by linarith
    rw [h2pi_sub] at hcos
    have : Real.cos (π / 4) < Real.cos (2 * π - x) :=
      Real.cos_lt_cos_of_nonneg_of_le_pi (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos]) (by linarith)
    have : Real.cos (π / 4) = Real.sqrt 2 / 2 := Real.cos_pi_div_four
    linarith
