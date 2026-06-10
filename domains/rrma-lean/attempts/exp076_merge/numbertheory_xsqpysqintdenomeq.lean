import Mathlib
set_option maxHeartbeats 16000000
open BigOperators Real Nat Topology Rat

theorem numbertheory_xsqpysqintdenomeq (x y : ℚ) (h₀ : (x ^ 2 + y ^ 2).den = 1) : x.den = y.den := by
  set n := (x ^ 2 + y ^ 2).num with hn_def
  have hn : (n : ℚ) = x ^ 2 + y ^ 2 := Rat.coe_int_num_of_den_eq_one h₀
  have hxd : (x.den : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp x.den_pos)
  have hyd : (y.den : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp y.den_pos)
  -- Key rational identity
  have key_rat : (n : ℚ) * (x.den : ℚ) ^ 2 * (y.den : ℚ) ^ 2 =
      (x.num : ℚ) ^ 2 * (y.den : ℚ) ^ 2 + (y.num : ℚ) ^ 2 * (x.den : ℚ) ^ 2 := by
    have : (n : ℚ) = ((x.num : ℚ) / x.den) ^ 2 + ((y.num : ℚ) / y.den) ^ 2 := by
      rw [hn, x.num_div_den, y.num_div_den]
    rw [this]
    field_simp
  -- Cast to ℤ
  have key_int : n * (x.den : ℤ) ^ 2 * (y.den : ℤ) ^ 2 =
      x.num ^ 2 * (y.den : ℤ) ^ 2 + y.num ^ 2 * (x.den : ℤ) ^ 2 := by
    exact_mod_cast key_rat
  -- xd^2 divides xn^2 * yd^2
  have hxd2_dvd_first : (x.den : ℤ) ^ 2 ∣ x.num ^ 2 * (y.den : ℤ) ^ 2 := by
    have h1 : (x.den : ℤ) ^ 2 ∣ n * (x.den : ℤ) ^ 2 * (y.den : ℤ) ^ 2 :=
      ⟨n * (y.den : ℤ) ^ 2, by ring⟩
    have h2 : (x.den : ℤ) ^ 2 ∣ y.num ^ 2 * (x.den : ℤ) ^ 2 :=
      ⟨y.num ^ 2, by ring⟩
    rw [key_int] at h1
    have h1' : (x.den : ℤ) ^ 2 ∣ y.num ^ 2 * (x.den : ℤ) ^ 2 + x.num ^ 2 * (y.den : ℤ) ^ 2 := by
      rwa [add_comm] at h1
    exact (dvd_add_right h2).mp h1'
  -- Coprimality
  have hx_cop : IsCoprime (x.num) ((x.den : ℤ)) := by
    rw [Int.isCoprime_iff_nat_coprime]; exact x.reduced
  have hx_cop2 : IsCoprime (x.num ^ 2) ((x.den : ℤ) ^ 2) := hx_cop.pow
  have hxd2_dvd_yd2 : (x.den : ℤ) ^ 2 ∣ (y.den : ℤ) ^ 2 :=
    hx_cop2.symm.dvd_of_dvd_mul_left hxd2_dvd_first
  -- Similarly yd^2 | xd^2
  have hyd2_dvd_second : (y.den : ℤ) ^ 2 ∣ y.num ^ 2 * (x.den : ℤ) ^ 2 := by
    have h1 : (y.den : ℤ) ^ 2 ∣ n * (x.den : ℤ) ^ 2 * (y.den : ℤ) ^ 2 :=
      ⟨n * (x.den : ℤ) ^ 2, by ring⟩
    have h2 : (y.den : ℤ) ^ 2 ∣ x.num ^ 2 * (y.den : ℤ) ^ 2 :=
      ⟨x.num ^ 2, by ring⟩
    rw [key_int] at h1
    have h1' : (y.den : ℤ) ^ 2 ∣ y.num ^ 2 * (x.den : ℤ) ^ 2 + x.num ^ 2 * (y.den : ℤ) ^ 2 := by
      rwa [add_comm] at h1
    exact (dvd_add_left h2).mp h1'
  have hy_cop : IsCoprime (y.num) ((y.den : ℤ)) := by
    rw [Int.isCoprime_iff_nat_coprime]; exact y.reduced
  have hy_cop2 : IsCoprime (y.num ^ 2) ((y.den : ℤ) ^ 2) := hy_cop.pow
  have hyd2_dvd_xd2 : (y.den : ℤ) ^ 2 ∣ (x.den : ℤ) ^ 2 :=
    hy_cop2.symm.dvd_of_dvd_mul_left hyd2_dvd_second
  -- xd = yd
  have hxd2_eq_yd2 : (x.den : ℤ) ^ 2 = (y.den : ℤ) ^ 2 :=
    le_antisymm
      (Int.le_of_dvd (by positivity) hxd2_dvd_yd2)
      (Int.le_of_dvd (by positivity) hyd2_dvd_xd2)
  have : (x.den : ℤ) = (y.den : ℤ) := by
    nlinarith [sq_nonneg ((x.den : ℤ) - (y.den : ℤ)),
               Int.natCast_pos.mpr x.den_pos, Int.natCast_pos.mpr y.den_pos]
  exact Nat.cast_injective this
