import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2010_p10 (p q : ℝ) (a : ℕ → ℝ) (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p) (h₂ : a 2 = 9) (h₃ : a 3 = 3 * p - q) (h₄ : a 4 = 3 * p + q) : a 2010 = 8041 := by
  have hd12 := h₀ 1; have hd23 := h₀ 2
  have hp : p = 5 := by linarith
  have ha0 : a 0 = 1 := by have := h₀ 0; linarith
  have key : ∀ n, a n = 1 + 4 * (n : ℝ) := by
    suffices ∀ n, a n = 1 + 4 * (n : ℝ) ∧ a (n + 1) = 1 + 4 * ((n : ℝ) + 1) by
      exact fun n => (this n).1
    intro n; induction n with
    | zero => constructor <;> (push_cast; linarith)
    | succ n ih =>
      obtain ⟨hn, hn1⟩ := ih
      have step := h₀ n
      constructor
      · push_cast at hn1 ⊢; linarith
      · push_cast at hn hn1 ⊢; linarith
  have := key 2010; push_cast at this; linarith
