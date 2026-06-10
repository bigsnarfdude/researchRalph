import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem aimeII_2001_p3 (x : ℕ → ℤ) (h₀ : x 1 = 211) (h₂ : x 2 = 375) (h₃ : x 3 = 420)
  (h₄ : x 4 = 523) (h₆ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4)) :
  x 531 + x 753 + x 975 = 898 := by
  have h5 : x 5 = 267 := by have := h₆ 5 (by omega); linarith
  have hx6 : x 6 = -211 := by have := h₆ 6 (by omega); linarith
  have hx7 : x 7 = -375 := by have := h₆ 7 (by omega); linarith
  have hx8 : x 8 = -420 := by have := h₆ 8 (by omega); linarith
  have hx9 : x 9 = -523 := by have := h₆ 9 (by omega); linarith
  have hx10 : x 10 = -267 := by have := h₆ 10 (by omega); linarith
  have hx11 : x 11 = 211 := by have := h₆ 11 (by omega); linarith
  have hx12 : x 12 = 375 := by have := h₆ 12 (by omega); linarith
  have hx13 : x 13 = 420 := by have := h₆ 13 (by omega); linarith
  have hx14 : x 14 = 523 := by have := h₆ 14 (by omega); linarith
  -- Prove periodicity: ∀ n ≥ 1, x(n+10) = x(n)
  have hperiod : ∀ n, 1 ≤ n → x (n + 10) = x n := by
    intro n hn
    induction n using Nat.strongRecOn with
    | _ n ih =>
      match n, hn with
      | 1, _ => linarith [hx11]
      | 2, _ => linarith [hx12]
      | 3, _ => linarith [hx13]
      | 4, _ => linarith [hx14]
      | n + 5, _ =>
        have hrec_shift := h₆ (n + 15) (by omega)
        have hrec_orig := h₆ (n + 5) (by omega)
        simp only [show n + 15 - 1 = n + 14 from by omega,
                    show n + 15 - 2 = n + 13 from by omega,
                    show n + 15 - 3 = n + 12 from by omega,
                    show n + 15 - 4 = n + 11 from by omega,
                    show n + 5 - 1 = n + 4 from by omega,
                    show n + 5 - 2 = n + 3 from by omega,
                    show n + 5 - 3 = n + 2 from by omega,
                    show n + 5 - 4 = n + 1 from by omega] at hrec_shift hrec_orig
        rw [show n + 5 + 10 = n + 15 from by omega]
        have ih1 := ih (n + 4) (by omega) (by omega)
        have ih2 := ih (n + 3) (by omega) (by omega)
        have ih3 := ih (n + 2) (by omega) (by omega)
        have ih4 := ih (n + 1) (by omega) (by omega)
        rw [show n + 4 + 10 = n + 14 from by omega] at ih1
        rw [show n + 3 + 10 = n + 13 from by omega] at ih2
        rw [show n + 2 + 10 = n + 12 from by omega] at ih3
        rw [show n + 1 + 10 = n + 11 from by omega] at ih4
        linarith
  -- Periodicity for multiples
  have hmul : ∀ (k m : ℕ), 1 ≤ m → x (m + 10 * k) = x m := by
    intro k; induction k with
    | zero => simp
    | succ k ihk =>
      intro m hm
      rw [show m + 10 * (k + 1) = (m + 10 * k) + 10 from by ring]
      rw [hperiod (m + 10 * k) (by omega)]
      exact ihk m hm
  have h531 : x 531 = x 1 := by convert hmul 53 1 (by omega) using 2
  have h753 : x 753 = x 3 := by convert hmul 75 3 (by omega) using 2
  have h975 : x 975 = x 5 := by convert hmul 97 5 (by omega) using 2
  linarith
