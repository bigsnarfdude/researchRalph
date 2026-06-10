import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_530 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₁ : (n : ℝ) / k < 6)
  (h₂ : (5 : ℝ) < n / k) : 22 ≤ Nat.lcm n k / Nat.gcd n k := by
  have hk_pos : (0 : ℝ) < k := Nat.cast_pos.mpr h₀.2
  have h5k : 5 * k < n := by exact_mod_cast (lt_div_iff₀ hk_pos).mp h₂
  have h6k : n < 6 * k := by exact_mod_cast (div_lt_iff₀ hk_pos).mp h₁
  set g := Nat.gcd n k with hg_def
  have hg_pos : 0 < g := Nat.gcd_pos_of_pos_left k h₀.1
  set a := n / g
  set b := k / g
  have hna : a * g = n := Nat.div_mul_cancel (Nat.gcd_dvd_left n k)
  have hkb : b * g = k := Nat.div_mul_cancel (Nat.gcd_dvd_right n k)
  have ha_pos : 0 < a := by rcases Nat.eq_zero_or_pos a with h | h; simp [h] at hna; omega; exact h
  have hb_pos : 0 < b := by rcases Nat.eq_zero_or_pos b with h | h; simp [h] at hkb; omega; exact h
  have h5b : 5 * b < a := by
    nlinarith [show 5 * (b * g) < a * g from by nlinarith [hna.symm, hkb.symm]]
  have h6b : a < 6 * b := by
    nlinarith [show a * g < 6 * (b * g) from by nlinarith [hna.symm, hkb.symm]]
  have hlcm_eq : Nat.lcm n k / g = a * b := by
    have hlcm_gcd : Nat.lcm n k * g = n * k := Nat.lcm_mul_gcd n k
    have h1 : n * k = a * b * g * g := by nlinarith [hna.symm, hkb.symm]
    have hlcm_val : Nat.lcm n k = a * b * g := by nlinarith
    rw [hlcm_val, Nat.mul_div_cancel _ hg_pos]
  rw [hlcm_eq]
  have hb2 : 2 ≤ b := by omega
  rcases le_or_gt 3 b with hb3 | hb3
  · nlinarith
  · -- b < 3 and b ≥ 2, so b = 2
    have hbeq : b = 2 := by omega
    -- a > 10 and a < 12, so a = 11
    have haeq : a = 11 := by omega
    rw [hbeq, haeq]
