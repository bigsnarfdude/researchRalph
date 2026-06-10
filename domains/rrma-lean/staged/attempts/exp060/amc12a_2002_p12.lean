import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2002_p12 (f : ℝ → ℝ) (k : ℝ) (a b : ℕ) (h₀ : ∀ x, f x = x ^ 2 - 63 * x + k)
  (h₁ : f a = 0 ∧ f b = 0) (h₂ : a ≠ b) (h₃ : Nat.Prime a ∧ Nat.Prime b) : k = 122 := by
  have ha := h₁.1; have hb := h₁.2
  simp [h₀] at ha hb
  have hab_sum : (a : ℝ) + b = 63 := by
    have : ((a : ℝ) - b) * ((a : ℝ) + b - 63) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · exact absurd (Nat.cast_injective (by linarith : (a : ℝ) = b)) h₂
    · linarith
  have hab_nat : a + b = 63 := by exact_mod_cast hab_sum
  have : a = 2 ∨ b = 2 := by
    by_contra h; push_neg at h
    have haodd := (Nat.Prime.eq_two_or_odd h₃.1).resolve_left h.1
    have hbodd := (Nat.Prime.eq_two_or_odd h₃.2).resolve_left h.2
    omega
  rcases this with rfl | rfl
  · push_cast at ha; linarith
  · push_cast at hb; linarith
