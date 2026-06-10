import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p11 (a b : ℕ) (h₀ : Nat.Prime a) (h₁ : Nat.Prime b) (h₂ : Nat.Prime (a + b))
  (h₃ : Nat.Prime (a - b)) : Nat.Prime (a + b + (a - b + (a + b))) := by
  have hab : b < a := by
    by_contra h
    push_neg at h
    have : a - b = 0 := Nat.sub_eq_zero_of_le h
    rw [this] at h₃
    exact Nat.not_prime_zero h₃
  have hb2 : b = 2 := by
    by_contra hb2
    have ha2 : a ≠ 2 := by
      intro ha2; subst ha2
      have := h₁.two_le
      omega
    have hao := h₀.eq_two_or_odd.resolve_left ha2
    have hbo := h₁.eq_two_or_odd.resolve_left hb2
    have hab_even : 2 ∣ a + b := by omega
    have hab_ge : a + b ≥ 4 := by have := h₀.two_le; have := h₁.two_le; omega
    exact (Nat.Prime.eq_one_or_self_of_dvd h₂ 2 hab_even).elim (by omega) (by omega)
  subst hb2
  have ha_gt : a > 2 := by omega
  have ha_le : a ≤ 7 := by
    by_contra h
    push_neg at h
    have : a % 3 = 0 ∨ a % 3 = 1 ∨ a % 3 = 2 := by omega
    rcases this with h0 | h1 | h2
    · have : 3 ∣ a := by omega
      exact (h₀.eq_one_or_self_of_dvd 3 this).elim (by omega) (by omega)
    · have : 3 ∣ (a + 2) := by omega
      exact (h₂.eq_one_or_self_of_dvd 3 this).elim (by omega) (by omega)
    · have : 3 ∣ (a - 2) := by omega
      exact (h₃.eq_one_or_self_of_dvd 3 this).elim (by omega) (by omega)
  interval_cases a <;> norm_num at *
