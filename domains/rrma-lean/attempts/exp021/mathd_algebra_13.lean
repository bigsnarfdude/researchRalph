import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- Partial fractions: 4x/(x²-8x+15) = a/(x-3) + b/(x-5)
-- Specialize at x=0 and x=1, solve linear system
theorem mathd_algebra_13 (a b : ℝ)
  (h₀ : ∀ x, x - 3 ≠ 0 ∧ x - 5 ≠ 0 → 4 * x / (x ^ 2 - 8 * x + 15) = a / (x - 3) + b / (x - 5)) :
  a = -6 ∧ b = 10 := by
  have h1 := h₀ 0 ⟨by norm_num, by norm_num⟩
  have h2 := h₀ 1 ⟨by norm_num, by norm_num⟩
  constructor <;> {
    field_simp at h1 h2
    linarith
  }
