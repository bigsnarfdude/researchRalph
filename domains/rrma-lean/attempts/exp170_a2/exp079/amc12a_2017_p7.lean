import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12a_2017_p7 (f : ℕ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  -- f(odd) = f(odd-2) + 2 chain: f(2k+1) = f(1) + 2k = 2+2k
  have hstep : ∀ n, 3 ≤ n → Odd n → f n = f (n - 2) + 2 := by
    intro n hn hodd; exact h₂ n ⟨by omega, hodd⟩
  suffices h : ∀ k, f (2 * k + 1) = 2 + 2 * (k : ℝ) by
    have := h 1008; norm_num at this; linarith
  intro k
  induction k with
  | zero => simp [h₀]
  | succ m ih =>
    have : 3 ≤ 2 * (m + 1) + 1 := by omega
    have hodd : Odd (2 * (m + 1) + 1) := ⟨m + 1, by omega⟩
    rw [hstep _ this hodd, show 2 * (m + 1) + 1 - 2 = 2 * m + 1 from by omega, ih]
    push_cast; ring
