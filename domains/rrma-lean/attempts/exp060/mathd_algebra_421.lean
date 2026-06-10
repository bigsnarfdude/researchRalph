import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_421 (a b c d : ℝ) (h₀ : b = a ^ 2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a ^ 2 + a + 6) (h₂ : d = c ^ 2 + 4 * c + 6) (h₃ : d = 1 / 2 * c ^ 2 + c + 6)
  (h₄ : a < c) : c - a = 6 := by
  have ha : a ^ 2 + 6 * a = 0 := by nlinarith
  have hc : c ^ 2 + 6 * c = 0 := by nlinarith
  -- a*(a+6)=0 and c*(c+6)=0, so a,c ∈ {0,-6}
  -- a < c so a = -6, c = 0
  have : a = 0 ∨ a = -6 := by
    have : a * (a + 6) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · left; exact h
    · right; linarith
  have : c = 0 ∨ c = -6 := by
    have : c * (c + 6) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · left; exact h
    · right; linarith
  rcases ‹a = 0 ∨ a = -6› with rfl | rfl <;> rcases ‹c = 0 ∨ c = -6› with rfl | rfl <;> linarith
