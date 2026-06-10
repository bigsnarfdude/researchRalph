import Mathlib
set_option maxHeartbeats 16000000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_530 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₀ : (n : ℝ) / k < 6)
  (h₁ : (5 : ℝ) < n / k) : 22 ≤ Nat.lcm n k / Nat.gcd n k := by
  have hk_pos : (0 : ℝ) < k := by
    by_contra h; push_neg at h
    have hk0 : (k : ℝ) = 0 := le_antisymm h (Nat.cast_nonneg k)
    have : (5 : ℝ) < 0 := by
      calc (5 : ℝ) < n / k := h₁
        _ = n / 0 := by rw [hk0]
        _ = 0 := div_zero _
    linarith
  have hk_ne : (k : ℝ) ≠ 0 := ne_of_gt hk_pos
  have h5k_lt_n : 5 * (k : ℝ) < n := by
    have := (lt_div_iff₀ hk_pos).mp h₁; linarith
  have hn_lt_6k : (n : ℝ) < 6 * k := by rwa [div_lt_iff₀ hk_pos] at h₀
  have h5k_lt_n_nat : 5 * k < n := by exact_mod_cast h5k_lt_n
  have hn_lt_6k_nat : n < 6 * k := by exact_mod_cast hn_lt_6k
  have hn_pos : 0 < n := by omega
  have hk_pos_nat : 0 < k := by exact_mod_cast hk_pos
  set d := Nat.gcd n k with hd_def
  have hd_pos : 0 < d := Nat.pos_of_ne_zero (by intro h; have := Nat.eq_zero_of_gcd_eq_zero_left h; omega)
  set a := n / d with ha_def
  set b := k / d with hb_def
  have hdn : d ∣ n := Nat.gcd_dvd_left n k
  have hdk : d ∣ k := Nat.gcd_dvd_right n k
  have hn_eq : n = a * d := by rw [ha_def]; exact (Nat.div_mul_cancel hdn).symm
  have hk_eq : k = b * d := by rw [hb_def]; exact (Nat.div_mul_cancel hdk).symm
  have ha_pos : 0 < a := by rw [ha_def]; exact Nat.div_pos (Nat.le_of_dvd hn_pos hdn) hd_pos
  have hb_pos : 0 < b := by rw [hb_def]; exact Nat.div_pos (Nat.le_of_dvd hk_pos_nat hdk) hd_pos
  have hab_coprime : Nat.Coprime a b :=
    Nat.coprime_div_gcd_div_gcd (by omega)
  have h5b_lt_a : 5 * b < a := by nlinarith [hn_eq, hk_eq, h5k_lt_n_nat]
  have ha_lt_6b : a < 6 * b := by nlinarith [hn_eq, hk_eq, hn_lt_6k_nat]
  have hb_ge2 : 2 ≤ b := by
    by_contra h; push_neg at h; interval_cases b <;> omega
  have ha_ge : a ≥ 5 * b + 1 := by omega
  have hab_ge : 22 ≤ a * b := by nlinarith
  suffices h : Nat.lcm n k / Nat.gcd n k = a * b by linarith
  -- lcm n k = n * k / gcd n k by definition
  -- gcd n k = d
  -- We need: (n * k / d) / d = a * b
  show Nat.lcm n k / d = a * b
  rw [Nat.lcm, show Nat.gcd n k = d from rfl]
  -- goal: n * k / d / d = a * b
  rw [hn_eq, hk_eq]
  -- goal: a * d * (b * d) / d / d = a * b
  rw [show a * d * (b * d) = a * b * d * d by ring]
  rw [Nat.mul_div_cancel _ (by positivity : 0 < d)]
  rw [Nat.mul_div_cancel _ (by positivity : 0 < d)]
