import Mathlib
set_option maxHeartbeats 16000000

theorem imo_1966_p5 (x a : ℕ → ℝ) (h₀ : a 1 ≠ a 2) (h₁ : a 1 ≠ a 3) (h₂ : a 1 ≠ a 4)
  (h₃ : a 2 ≠ a 3) (h₄ : a 2 ≠ a 4) (h₅ : a 3 ≠ a 4) (h₆ : a 1 > a 2) (h₇ : a 2 > a 3)
  (h₈ : a 3 > a 4)
  (h₉ : abs (a 1 - a 2) * x 2 + abs (a 1 - a 3) * x 3 + abs (a 1 - a 4) * x 4 = 1)
  (h₁₀ : abs (a 2 - a 1) * x 1 + abs (a 2 - a 3) * x 3 + abs (a 2 - a 4) * x 4 = 1)
  (h₁₁ : abs (a 3 - a 1) * x 1 + abs (a 3 - a 2) * x 2 + abs (a 3 - a 4) * x 4 = 1)
  (h₁₂ : abs (a 4 - a 1) * x 1 + abs (a 4 - a 2) * x 2 + abs (a 4 - a 3) * x 3 = 1) :
  x 2 = 0 ∧ x 3 = 0 ∧ x 1 = 1 / abs (a 1 - a 4) ∧ x 4 = 1 / abs (a 1 - a 4) := by
  simp only [abs_of_pos (show a 1 - a 2 > 0 by linarith),
    abs_of_pos (show a 1 - a 3 > 0 by linarith),
    abs_of_pos (show a 1 - a 4 > 0 by linarith),
    abs_of_pos (show a 2 - a 3 > 0 by linarith),
    abs_of_pos (show a 2 - a 4 > 0 by linarith),
    abs_of_pos (show a 3 - a 4 > 0 by linarith),
    show abs (a 2 - a 1) = a 1 - a 2 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith),
    show abs (a 3 - a 1) = a 1 - a 3 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith),
    show abs (a 3 - a 2) = a 2 - a 3 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith),
    show abs (a 4 - a 1) = a 1 - a 4 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith),
    show abs (a 4 - a 2) = a 2 - a 4 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith),
    show abs (a 4 - a 3) = a 3 - a 4 from by rw [abs_sub_comm]; exact abs_of_pos (by linarith)
  ] at h₉ h₁₀ h₁₁ h₁₂ ⊢
  have eq1 : x 1 = x 2 + x 3 + x 4 := by
    have : (a 1 - a 2) * (x 2 - x 1 + x 3 + x 4) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h <;> linarith
  have eq2 : x 4 = x 1 + x 2 + x 3 := by
    have : (a 3 - a 4) * (x 1 + x 2 + x 3 - x 4) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h <;> linarith
  have hx23 : x 3 = x 2 := by
    have : (a 2 - a 3) * (x 3 - x 2) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h <;> linarith
  have hx2 : x 2 = 0 := by linarith
  have hx3 : x 3 = 0 := by linarith
  have hx14 : x 1 = x 4 := by linarith
  have had : (a 1 - a 4) ≠ 0 := by linarith
  have hx1_eq : x 1 = 1 / (a 1 - a 4) := by
    have : (a 1 - a 4) * x 1 = 1 := by nlinarith
    field_simp at this ⊢; linarith
  exact ⟨hx2, hx3, hx1_eq, by linarith⟩
