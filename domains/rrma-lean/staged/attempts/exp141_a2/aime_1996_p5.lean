import Mathlib

set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem aime_1996_p5 (a b c r s t : ℝ) (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x ^ 3 + 3 * x ^ 2 + 4 * x - 11) (h₁ : ∀ x, g x = x ^ 3 + r * x ^ 2 + s * x + t)
  (h₂ : f a = 0) (h₃ : f b = 0) (h₄ : f c = 0) (h₅ : g (a + b) = 0) (h₆ : g (b + c) = 0)
  (h₇ : g (c + a) = 0) (h₈ : List.Pairwise (· ≠ ·) [a, b, c]) : t = 23 := by
  have ha : a ^ 3 + 3 * a ^ 2 + 4 * a = 11 := by have := h₂; simp [h₀] at this; linarith
  have hb : b ^ 3 + 3 * b ^ 2 + 4 * b = 11 := by have := h₃; simp [h₀] at this; linarith
  have hc : c ^ 3 + 3 * c ^ 2 + 4 * c = 11 := by have := h₄; simp [h₀] at this; linarith
  have gab : (a + b) ^ 3 + r * (a + b) ^ 2 + s * (a + b) + t = 0 := by
    have := h₅; simp [h₁] at this; linarith
  have gbc : (b + c) ^ 3 + r * (b + c) ^ 2 + s * (b + c) + t = 0 := by
    have := h₆; simp [h₁] at this; linarith
  have gca : (c + a) ^ 3 + r * (c + a) ^ 2 + s * (c + a) + t = 0 := by
    have := h₇; simp [h₁] at this; linarith
  have hab : a ≠ b := by intro h; subst h; simp [List.pairwise_cons] at h₈
  have hbc : b ≠ c := by intro h; subst h; simp [List.pairwise_cons] at h₈
  have hac : a ≠ c := by intro h; subst h; simp [List.pairwise_cons] at h₈
  -- a+b+c = -3
  have h_sum : a + b + c = -3 := by
    have h1 : (a - b) * (a ^ 2 + a * b + b ^ 2 + 3 * a + 3 * b + 4) = 0 := by nlinarith
    have h2 : (b - c) * (b ^ 2 + b * c + c ^ 2 + 3 * b + 3 * c + 4) = 0 := by nlinarith
    have := (mul_eq_zero.mp h1).resolve_left (sub_ne_zero.mpr hab)
    have := (mul_eq_zero.mp h2).resolve_left (sub_ne_zero.mpr hbc)
    have h3 : (a - c) * (a + b + c + 3) = 0 := by nlinarith
    linarith [(mul_eq_zero.mp h3).resolve_left (sub_ne_zero.mpr hac)]
  -- r = 6
  have h_r : r = 6 := by
    have g1 : (a - c) * ((a+b)^2+(a+b)*(b+c)+(b+c)^2 + r*((a+b)+(b+c)) + s) = 0 := by nlinarith
    have g2 : (b - a) * ((b+c)^2+(b+c)*(c+a)+(c+a)^2 + r*((b+c)+(c+a)) + s) = 0 := by nlinarith
    have := (mul_eq_zero.mp g1).resolve_left (sub_ne_zero.mpr hac)
    have := (mul_eq_zero.mp g2).resolve_left (sub_ne_zero.mpr (Ne.symm hab))
    have g3 : (b - c) * (2*(a+b+c) + r) = 0 := by nlinarith
    linarith [(mul_eq_zero.mp g3).resolve_left (sub_ne_zero.mpr hbc), h_sum]
  -- ab+bc+ca = 4
  have h_S : a * b + b * c + a * c = 4 := by
    have eq1 : a ^ 2 + a * b + b ^ 2 + 3 * a + 3 * b + 4 = 0 := by
      have : (a - b) * (a ^ 2 + a * b + b ^ 2 + 3 * a + 3 * b + 4) = 0 := by nlinarith
      exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hab)
    have eq2 : b ^ 2 + b * c + c ^ 2 + 3 * b + 3 * c + 4 = 0 := by
      have : (b - c) * (b ^ 2 + b * c + c ^ 2 + 3 * b + 3 * c + 4) = 0 := by nlinarith
      exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hbc)
    have eq3 : a ^ 2 + a * c + c ^ 2 + 3 * a + 3 * c + 4 = 0 := by
      have : (a - c) * (a ^ 2 + a * c + c ^ 2 + 3 * a + 3 * c + 4) = 0 := by nlinarith
      exact (mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr hac)
    nlinarith [h_sum]
  -- abc = 11
  have h_P : a * b * c = 11 := by nlinarith [h_sum, h_S]
  -- t = (a+3)(b+3)(c+3) = abc + 3(ab+bc+ca) + 9(a+b+c) + 27 = 11+12-27+27 = 23
  -- From g equations with r=6: can express t. Use nlinarith.
  nlinarith [h_sum, h_r, h_S, h_P, gab, gbc, gca,
             sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (a - c)]
