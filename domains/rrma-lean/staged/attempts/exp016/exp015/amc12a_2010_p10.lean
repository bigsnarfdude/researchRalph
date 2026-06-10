import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p10 (p q : ℝ) (a : ℕ → ℝ) (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p) (h₂ : a 2 = 9) (h₃ : a 3 = 3 * p - q) (h₄ : a 4 = 3 * p + q) : a 2010 = 8041 := by
  have hd1 := h₀ 1
  have hd2 := h₀ 2
  have hp : p = 5 := by linarith
  have hd0 := h₀ 0
  have ha0 : a 0 = 1 := by linarith
  have hstep : ∀ n, a (n + 1) - a n = 4 := by
    intro n
    induction n with
    | zero => linarith
    | succ k ih =>
      have hk := h₀ k
      linarith
  suffices h : ∀ n, a n = 1 + 4 * (n : ℝ) by
    have := h 2010; norm_num at this; linarith
  intro n
  induction n with
  | zero => simp; linarith
  | succ k ih =>
    have := hstep k
    push_cast
    linarith
