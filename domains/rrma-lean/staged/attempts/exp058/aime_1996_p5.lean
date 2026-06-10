import Mathlib
set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem aime_1996_p5 (a b c r s t : ℝ) (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x ^ 3 + 3 * x ^ 2 + 4 * x - 11)
  (h₁ : ∀ x, g x = x ^ 3 + r * x ^ 2 + s * x + t)
  (h₂ : f a = 0) (h₃ : f b = 0) (h₄ : f c = 0)
  (h₅ : g (a + b) = 0) (h₆ : g (b + c) = 0) (h₇ : g (c + a) = 0)
  (h₈ : List.Pairwise (· ≠ ·) [a, b, c]) : t = 23 := by
  rw [List.pairwise_cons] at h₈; obtain ⟨hall, htl⟩ := h₈
  rw [List.pairwise_cons] at htl; obtain ⟨hall2, _⟩ := htl
  have hab : a ≠ b := by apply hall; simp
  have hac : a ≠ c := by apply hall; simp
  have hbc : b ≠ c := by apply hall2; simp
  have ha : a ^ 3 + 3 * a ^ 2 + 4 * a - 11 = 0 := by have := h₂; rw [h₀] at this; linarith
  have hb : b ^ 3 + 3 * b ^ 2 + 4 * b - 11 = 0 := by have := h₃; rw [h₀] at this; linarith
  have hc : c ^ 3 + 3 * c ^ 2 + 4 * c - 11 = 0 := by have := h₄; rw [h₀] at this; linarith
  have hg1 : (a+b)^3 + r*(a+b)^2 + s*(a+b) + t = 0 := by have := h₅; rw [h₁] at this; linarith
  have hg2 : (b+c)^3 + r*(b+c)^2 + s*(b+c) + t = 0 := by have := h₆; rw [h₁] at this; linarith
  have hg3 : (c+a)^3 + r*(c+a)^2 + s*(c+a) + t = 0 := by have := h₇; rw [h₁] at this; linarith
  -- Step 1: a+b+c = -3
  have hab' : a^2+a*b+b^2+3*a+3*b+4 = 0 := by
    have hfact : (a-b) * (a^2+a*b+b^2+3*a+3*b+4) = 0 := by nlinarith
    cases mul_eq_zero.mp hfact with
    | inl h => exact absurd (sub_eq_zero.mp h) hab
    | inr h => linarith
  have hac' : a^2+a*c+c^2+3*a+3*c+4 = 0 := by
    have hfact : (a-c) * (a^2+a*c+c^2+3*a+3*c+4) = 0 := by nlinarith
    cases mul_eq_zero.mp hfact with
    | inl h => exact absurd (sub_eq_zero.mp h) hac
    | inr h => linarith
  have hsum : a + b + c = -3 := by
    have hfact : (b-c) * (a+b+c+3) = 0 := by nlinarith
    cases mul_eq_zero.mp hfact with
    | inl h => exact absurd (sub_eq_zero.mp h) hbc
    | inr h => linarith
  -- Step 2: r = 6
  have hr : r = 6 := by
    have hfact1 : (a-c)*((a+b)^2+(a+b)*(b+c)+(b+c)^2+r*((a+b)+(b+c))+s) = 0 := by nlinarith
    have hI : (a+b)^2+(a+b)*(b+c)+(b+c)^2+r*((a+b)+(b+c))+s = 0 := by
      cases mul_eq_zero.mp hfact1 with
      | inl h => exact absurd (sub_eq_zero.mp h) hac
      | inr h => exact h
    have hfact2 : (b-a)*((b+c)^2+(b+c)*(c+a)+(c+a)^2+r*((b+c)+(c+a))+s) = 0 := by nlinarith
    have hII : (b+c)^2+(b+c)*(c+a)+(c+a)^2+r*((b+c)+(c+a))+s = 0 := by
      cases mul_eq_zero.mp hfact2 with
      | inl h => exact absurd (sub_eq_zero.mp h).symm hab
      | inr h => exact h
    have hfact3 : (b-c)*(2*(a+b+c)+r) = 0 := by nlinarith
    cases mul_eq_zero.mp hfact3 with
    | inl h => exact absurd (sub_eq_zero.mp h) hbc
    | inr h => linarith
  -- Step 3: b²+bc+c²+3b+3c+4 = 0
  have hbc' : b^2+b*c+c^2+3*b+3*c+4 = 0 := by
    have hfact : (b-c)*(b^2+b*c+c^2+3*b+3*c+4) = 0 := by nlinarith
    cases mul_eq_zero.mp hfact with
    | inl h => exact absurd (sub_eq_zero.mp h) hbc
    | inr h => linarith
  -- Step 4: s = 13
  have hs : s = 13 := by
    have hfact : (b-c)*((a+b)^2+(a+b)*(c+a)+(c+a)^2+6*((a+b)+(c+a))+s) = 0 := by nlinarith
    have h4 : (a+b)^2+(a+b)*(c+a)+(c+a)^2+6*((a+b)+(c+a))+s = 0 := by
      cases mul_eq_zero.mp hfact with
      | inl h => exact absurd (sub_eq_zero.mp h) hbc
      | inr h => exact h
    nlinarith [hbc', hsum]
  -- Step 5: t = 23
  nlinarith [hc, hsum, hr, hs]
