import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2017_p7 (f : ℕ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  have key : ∀ n : ℕ, 1 ≤ n → f n = ↑n + 1 := by
    intro n
    induction n using Nat.strongRecOn with
    | _ n ih =>
      intro hn
      match n, hn with
      | 1, _ => norm_num; exact h₀
      | n + 2, hn =>
        by_cases he : Even (n + 2)
        · have := h₁ (n + 2) ⟨by omega, he⟩
          rw [show n + 2 - 1 = n + 1 from by omega] at this
          rw [this, ih (n + 1) (by omega) (by omega)]
          push_cast; ring
        · have ho : Odd (n + 2) := Nat.not_even_iff_odd.mp he
          have hge : 1 ≤ n := by
            by_contra h
            push_neg at h
            interval_cases n <;> simp [Nat.odd_iff] at ho
          have := h₂ (n + 2) ⟨by omega, ho⟩
          rw [show n + 2 - 2 = n from by omega] at this
          rw [this, ih n (by omega) hge]
          push_cast; ring
  have := key 2017 (by omega)
  push_cast at this; linarith
