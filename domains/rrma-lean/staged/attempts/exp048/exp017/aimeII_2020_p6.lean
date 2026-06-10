import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem aimeII_2020_p6 (t : ℕ → ℚ) (h₀ : t 1 = 20) (h₁ : t 2 = 21)
  (h₂ : ∀ n ≥ 3, t n = (5 * t (n - 1) + 1) / (25 * t (n - 2))) :
  ↑(t 2020).den + (t 2020).num = 626 := by
  have ht3 : t 3 = 53 / 250 := by
    have := h₂ 3 (by omega); rw [h₁, h₀] at this; rw [this]; norm_num
  have ht4 : t 4 = 103 / 26250 := by
    have := h₂ 4 (by omega); rw [ht3, h₁] at this; rw [this]; norm_num
  have ht5 : t 5 = 101 / 525 := by
    have := h₂ 5 (by omega); rw [ht4, ht3] at this; rw [this]; norm_num
  have ht6 : t 6 = 20 := by
    have := h₂ 6 (by omega); rw [ht5, ht4] at this; rw [this]; norm_num
  have ht7 : t 7 = 21 := by
    have := h₂ 7 (by omega); rw [ht6, ht5] at this; rw [this]; norm_num
  have hperiod : ∀ n, 1 ≤ n → t (n + 5) = t n := by
    intro n hn
    induction n using Nat.strongRecOn with
    | _ n ih =>
      match n, hn with
      | 1, _ => linarith [ht6]
      | 2, _ => linarith [ht7]
      | n + 3, _ =>
        have hrec_s := h₂ (n + 8) (by omega)
        have hrec_o := h₂ (n + 3) (by omega)
        simp only [show n + 8 - 1 = n + 7 from by omega,
                    show n + 8 - 2 = n + 6 from by omega,
                    show n + 3 - 1 = n + 2 from by omega,
                    show n + 3 - 2 = n + 1 from by omega] at hrec_s hrec_o
        rw [show n + 3 + 5 = n + 8 from by omega]
        have ih1 := ih (n + 2) (by omega) (by omega)
        have ih2 := ih (n + 1) (by omega) (by omega)
        rw [show n + 2 + 5 = n + 7 from by omega] at ih1
        rw [show n + 1 + 5 = n + 6 from by omega] at ih2
        rw [ih1, ih2] at hrec_s; linarith
  have hmul : ∀ (k m : ℕ), 1 ≤ m → t (m + 5 * k) = t m := by
    intro k; induction k with
    | zero => simp
    | succ k ihk =>
      intro m hm
      rw [show m + 5 * (k + 1) = (m + 5 * k) + 5 from by ring]
      rw [hperiod (m + 5 * k) (by omega)]
      exact ihk m hm
  have h2020 : t 2020 = t 5 := by convert hmul 403 5 (by omega) using 2
  rw [h2020, ht5]; norm_num
