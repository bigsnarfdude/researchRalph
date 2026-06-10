import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p10 (p q : ℝ) (a : ℕ → ℝ) (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p) (h₂ : a 2 = 9) (h₃ : a 3 = 3 * p - q) (h₄ : a 4 = 3 * p + q) : a 2010 = 8041 := by
  have hp : p = 5 := by linarith [h₀ 1, h₀ 2]
  have hq : q = 2 := by linarith [h₀ 1]
  have ha0 : a 0 = 1 := by linarith [h₀ 0]
  have hstep : ∀ n, a (n + 1) - a n = 4 := by
    intro n
    induction n with
    | zero => linarith
    | succ k ih => linarith [h₀ k]
  have hmain : ∀ n, a n = 1 + 4 * (n : ℝ) := by
    intro n
    induction n with
    | zero => simp [ha0]
    | succ k ih =>
      have := hstep k
      push_cast; linarith
  have := hmain 2010
  push_cast at this; linarith
