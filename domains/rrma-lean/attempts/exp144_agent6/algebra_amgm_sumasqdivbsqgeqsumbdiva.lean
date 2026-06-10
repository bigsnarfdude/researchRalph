import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
-- a²/b² + b²/c² + c²/a² - (b/a + c/b + a/c) = ((a/b-b/c)²+(b/c-c/a)²+(c/a-a/b)²)/2
theorem algebra_amgm_sumasqdivbsqgeqsumbdiva (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) :
  a ^ 2 / b ^ 2 + b ^ 2 / c ^ 2 + c ^ 2 / a ^ 2 ≥ b / a + c / b + a / c := by
  have ha := h₀.1; have hb := h₀.2.1; have hc := h₀.2.2
  rw [ge_iff_le, ← sub_nonneg]
  have key : a^2/b^2 + b^2/c^2 + c^2/a^2 - (b/a + c/b + a/c) =
    ((a/b - b/c)^2 + (b/c - c/a)^2 + (c/a - a/b)^2) / 2 := by
    field_simp
    ring
  rw [key]
  positivity
