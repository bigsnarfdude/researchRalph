import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem amc12b_2002_p11 (a b : ℕ) (h₀ : Nat.Prime a) (h₁ : Nat.Prime b) (h₂ : Nat.Prime (a + b))
  (h₃ : Nat.Prime (a - b)) : Nat.Prime (a + b + (a - b + (a + b))) := by
  have hb2 : b = 2 := by
    by_contra hb
    have hb_ne : b ≠ 2 := hb
    have hb_odd : b % 2 = 1 := by
      have := h₁.eq_two_or_odd
      rcases this with rfl | hodd
      · exact absurd rfl hb_ne
      · exact hodd
    by_cases ha : a = 2
    · have hb_ge : b ≥ 3 := by have := h₁.two_le; omega
      have : a - b = 0 := by omega
      rw [this] at h₃; exact Nat.not_prime_zero h₃
    · have ha_odd : a % 2 = 1 := by
        have := h₀.eq_two_or_odd; rcases this with rfl | hodd
        · exact absurd rfl ha; · exact hodd
      have hab_even : (a + b) % 2 = 0 := by omega
      have hab_ge : a + b ≥ 6 := by have := h₀.two_le; have := h₁.two_le; omega
      have hab_eq_2 : a + b = 2 := by
        have := h₂.eq_two_or_odd; rcases this with h | h; · exact h; · omega
      omega
  subst hb2
  have ha_ge : a ≥ 5 := by
    have := h₃; have h_sub_ge := this.two_le; have := h₀.two_le
    have := h₀.eq_two_or_odd; rcases this with rfl | hodd
    · exact absurd h₃ (by decide)
    · omega
  have ha_le : a ≤ 7 := by
    by_contra h; push_neg at h; have ha_ge8 : a ≥ 8 := h
    have hmod : a % 3 = 0 ∨ a % 3 = 1 ∨ a % 3 = 2 := by omega
    rcases hmod with hm | hm | hm
    · have : 3 ∣ a := by omega; have := h₀.eq_one_or_self_of_dvd 3 this; omega
    · have : 3 ∣ (a + 2) := by omega; have := h₂.eq_one_or_self_of_dvd 3 this; omega
    · have : 3 ∣ (a - 2) := by omega; have := h₃.eq_one_or_self_of_dvd 3 this; omega
  interval_cases a <;> simp_all (config := { decide := true }) <;> norm_num [Nat.Prime] at *
