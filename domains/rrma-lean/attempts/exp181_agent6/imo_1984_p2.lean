import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat

theorem imo_1984_p2 (a b : ℤ) (h₀ : 0 < a ∧ 0 < b) (h₁ : ¬7 ∣ a) (h₂ : ¬7 ∣ b) (h₃ : ¬7 ∣ a + b)
  (h₄ : 7 ^ 7 ∣ (a + b) ^ 7 - a ^ 7 - b ^ 7) : 19 ≤ a + b := by
  have identity : (a + b) ^ 7 - a ^ 7 - b ^ 7 =
    7 * a * b * (a + b) * (a ^ 2 + a * b + b ^ 2) ^ 2 := by ring
  rw [identity] at h₄
  have h5 : 7 ^ 6 ∣ a * b * (a + b) * (a ^ 2 + a * b + b ^ 2) ^ 2 := by
    rwa [show (7 : ℤ) ^ 7 = 7 * 7 ^ 6 from by ring,
         show 7 * a * b * (a + b) * (a ^ 2 + a * b + b ^ 2) ^ 2 =
              7 * (a * b * (a + b) * (a ^ 2 + a * b + b ^ 2) ^ 2) from by ring,
         Int.mul_dvd_mul_iff_left (by norm_num : (7 : ℤ) ≠ 0)] at h₄
  have hp7 : Prime (7 : ℤ) := by decide
  have hc1 : ¬(7 : ℤ) ∣ a * b * (a + b) := by
    intro h; rcases hp7.dvd_or_dvd h with hab | hc
    · rcases hp7.dvd_or_dvd hab with ha | hb; exact h₁ ha; exact h₂ hb
    · exact h₃ hc
  have hcop : IsCoprime ((7 : ℤ) ^ 6) (a * b * (a + b)) :=
    IsCoprime.pow_left (hp7.coprime_iff_not_dvd.mpr hc1)
  have h5' : (7 : ℤ) ^ 6 ∣ (a * b * (a + b)) * (a ^ 2 + a * b + b ^ 2) ^ 2 := by
    rwa [show a * b * (a + b) * (a ^ 2 + a * b + b ^ 2) ^ 2 =
        (a * b * (a + b)) * (a ^ 2 + a * b + b ^ 2) ^ 2 from by ring] at h5
  have h6 : (7 : ℤ) ^ 6 ∣ (a ^ 2 + a * b + b ^ 2) ^ 2 :=
    (IsCoprime.dvd_of_dvd_mul_left hcop) h5'
  have hY_pos : 0 < a ^ 2 + a * b + b ^ 2 := by nlinarith [h₀.1, h₀.2]
  have hY2_pos : 0 < (a ^ 2 + a * b + b ^ 2) ^ 2 := by positivity
  have hY2_ge : (7 : ℤ) ^ 6 ≤ (a ^ 2 + a * b + b ^ 2) ^ 2 :=
    Int.le_of_dvd hY2_pos h6
  have h8 : (343 : ℤ) ≤ a ^ 2 + a * b + b ^ 2 := by
    nlinarith [sq_nonneg (a ^ 2 + a * b + b ^ 2 - 343)]
  have hab_pos : 1 ≤ a * b := by nlinarith [h₀.1, h₀.2]
  have h9 : (344 : ℤ) ≤ (a + b) ^ 2 := by nlinarith
  have h10 : 0 < a + b := by linarith [h₀.1, h₀.2]
  nlinarith [sq_nonneg (a + b - 19)]
