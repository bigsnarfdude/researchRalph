import Mathlib
set_option maxHeartbeats 3200000
open BigOperators Real Nat Topology Rat
theorem aime_1996_p5 (a b c r s t : ℝ) (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x ^ 3 + 3 * x ^ 2 + 4 * x - 11) (h₁ : ∀ x, g x = x ^ 3 + r * x ^ 2 + s * x + t)
  (h₂ : f a = 0) (h₃ : f b = 0) (h₄ : f c = 0) (h₅ : g (a + b) = 0) (h₆ : g (b + c) = 0)
  (h₇ : g (c + a) = 0) (h₈ : List.Pairwise (· ≠ ·) [a, b, c]) : t = 23 := by
  simp [h₀] at h₂ h₃ h₄; simp [h₁] at h₅ h₆ h₇
  simp [List.pairwise_cons] at h₈
  have hab : a ≠ b := h₈.1.1
  have hac : a ≠ c := h₈.1.2
  have hbc : b ≠ c := h₈.2
  have hab_eq : a^2+a*b+b^2+3*a+3*b+4 = 0 := by
    have : (a-b)*(a^2+a*b+b^2+3*a+3*b+4) = 0 := by nlinarith
    exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hab)
  have hbc_eq : b^2+b*c+c^2+3*b+3*c+4 = 0 := by
    have : (b-c)*(b^2+b*c+c^2+3*b+3*c+4) = 0 := by nlinarith
    exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hbc)
  have hsum : a+b+c = -3 := by
    have : (a-c)*(a+b+c+3) = 0 := by nlinarith [hab_eq, hbc_eq]
    linarith [(mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hac)]
  have Pc : (r-6)*c^2 + (6*r-s-23)*c + (9*r-3*s+t-38) = 0 := by nlinarith
  have Pa : (r-6)*a^2 + (6*r-s-23)*a + (9*r-3*s+t-38) = 0 := by nlinarith
  have hr_ca : (r-6)*(c+a) + (6*r-s-23) = 0 := by
    have : (c-a)*((r-6)*(c+a) + (6*r-s-23)) = 0 := by nlinarith
    exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr (Ne.symm hac))
  have hr_ab : (r-6)*(a+b) + (6*r-s-23) = 0 := by
    have : (a-b)*((r-6)*(a+b) + (6*r-s-23)) = 0 := by nlinarith
    exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hab)
  have hr : r = 6 := by
    have : (r-6)*(c-b) = 0 := by linarith
    linarith [(mul_eq_zero.mp this).resolve_right (sub_ne_zero.mpr (Ne.symm hbc))]
  have hs : s = 13 := by nlinarith [hr, hr_ab]
  nlinarith [hr, hs, Pc]
