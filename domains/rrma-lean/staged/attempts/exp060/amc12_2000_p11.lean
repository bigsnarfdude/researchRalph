import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :
    a / b + b / a - a * b = 2 := by
  have ha := h₀.1; have hb := h₀.2
  have : a / b + b / a = (a^2 + b^2) / (a * b) := by field_simp
  rw [this, h₁]
  have hab : a - b ≠ 0 := by
    intro h; have := h₁; rw [show a - b = 0 from h] at this
    simp at this; rcases this with rfl | rfl <;> simp_all
  field_simp; nlinarith [h₁]
