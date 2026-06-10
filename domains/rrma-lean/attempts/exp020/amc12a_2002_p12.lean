import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p12 (f : ℝ → ℝ) (k : ℝ) (a b : ℕ) (h₀ : ∀ x, f x = x ^ 2 - 63 * x + k)
  (h₁ : f a = 0 ∧ f b = 0) (h₂ : a ≠ b) (h₃ : Nat.Prime a ∧ Nat.Prime b) : k = 122 := by
  have ha := h₁.1; have hb := h₁.2
  rw [h₀] at ha hb
  -- (a:ℝ)^2 - 63*a + k = 0 and (b:ℝ)^2 - 63*b + k = 0
  -- Subtracting: a^2 - b^2 - 63(a-b) = 0
  have hdiff : ((a : ℝ) - b) * (a + b - 63) = 0 := by nlinarith
  have hab_ne : (a : ℝ) ≠ b := by exact_mod_cast h₂
  have hab : (a : ℝ) + b = 63 := by
    rcases mul_eq_zero.mp hdiff with h | h
    · exact absurd (sub_eq_zero.mp h) hab_ne
    · linarith
  have hab_nat : a + b = 63 := by exact_mod_cast hab
  -- Both primes, sum is 63 (odd). One must be 2 (the only even prime).
  have ha2_or_hb2 : a = 2 ∨ b = 2 := by
    by_contra hc; push_neg at hc
    have hao := (h₃.1.eq_two_or_odd.resolve_left hc.1)
    have hbo := (h₃.2.eq_two_or_odd.resolve_left hc.2)
    omega
  have hk : k = (a : ℝ) * b := by nlinarith
  rcases ha2_or_hb2 with rfl | rfl
  · have : b = 61 := by omega
    subst this; rw [hk]; push_cast; norm_num
  · have : a = 61 := by omega
    subst this; rw [hk]; push_cast; norm_num
