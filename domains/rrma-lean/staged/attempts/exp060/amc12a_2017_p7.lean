import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2017_p7 (f : ℕ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  suffices ∀ k, f (2 * k + 1) = 2 * (k : ℝ) + 2 by
    have := this 1008; push_cast at this ⊢; linarith
  intro k; induction k with
  | zero => simp [h₀]
  | succ k ih =>
    have hodd : 1 < 2 * (k + 1) + 1 ∧ Odd (2 * (k + 1) + 1) := by
      constructor; omega; exact ⟨k + 1, by omega⟩
    have := h₂ _ hodd
    simp only [show 2 * (k + 1) + 1 - 2 = 2 * k + 1 from by omega] at this
    rw [ih] at this; push_cast at this ⊢; linarith
