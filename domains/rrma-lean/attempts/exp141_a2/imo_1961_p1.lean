import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem imo_1961_p1 (x y z a b : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x ≠ y) (h₂ : y ≠ z)
  (h₃ : z ≠ x) (h₄ : x + y + z = a) (h₅ : x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆ : x * y = z ^ 2) :
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
  have hx := h₀.1; have hy := h₀.2.1; have hz := h₀.2.2
  have ha_pos : 0 < a := by linarith
  have h_sum : x * y + y * z + x * z = z * a := by rw [h₆, ← h₄]; ring
  have ha2 : a ^ 2 = b ^ 2 + 2 * z * a := by nlinarith [h₅, h_sum, sq_nonneg a, sq_nonneg (x+y+z)]
  have h3key : 3 * b ^ 2 - a ^ 2 = (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 := by
    rw [← h₄, ← h₅]; ring
  have hxy_sq : (x - y) ^ 2 > 0 := by
    apply sq_pos_of_ne_zero; exact sub_ne_zero.mpr h₁
  refine ⟨ha_pos, ?_, ?_⟩
  · nlinarith
  · nlinarith [sq_nonneg (y - z), sq_nonneg (z - x)]
