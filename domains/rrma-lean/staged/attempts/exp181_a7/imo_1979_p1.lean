import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- IMO 1979 Q1: 1979 | p where p/q = ∑_{k=1}^{1319} (-1)^{k+1}/k

-- Key computational facts (verified by native_decide)
private lemma sum_num_div_1979 :
    (1979 : ℤ) ∣ (∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℚ) ^ (k + 1) * (1 / k)).num := by
  native_decide

private lemma sum_den_not_div_1979 :
    ¬ (1979 : ℕ) ∣ (∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℚ) ^ (k + 1) * (1 / k)).den := by
  native_decide

theorem imo_1979_p1 (p q : ℕ) (h₀ : 0 < q)
  (h₁ : (∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1) ^ (k + 1) * ((1 : ℝ) / k)) = p / q) : 1979 ∣ p := by
  set S := ∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℚ) ^ (k + 1) * ((1 : ℚ) / k)
  -- Cast ℚ sum to ℝ = the ℝ sum
  have hcast : (S : ℝ) = ∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℝ) ^ (k + 1) * (1 / k) := by
    simp only [Rat.cast_sum, Rat.cast_mul, Rat.cast_pow, Rat.cast_neg, Rat.cast_one,
      Rat.cast_div, Rat.cast_natCast, S]
  -- (S : ℝ) = p / q
  rw [← hcast] at h₁
  -- S = S.num / S.den as a ℚ fraction
  have hfrac := Rat.num_div_den S
  -- Cross multiply in ℝ: S.num * q = p * S.den
  have hq_ne : (q : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hden_ne : (S.den : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Rat.den_pos S).ne'
  have h_cross_r : (S.num : ℝ) * q = (p : ℝ) * S.den := by
    have hS_eq : (S : ℝ) = (S.num : ℝ) / (S.den : ℝ) := Rat.cast_def S
    rw [hS_eq] at h₁
    field_simp at h₁
    linarith
  -- Lift to ℤ: S.num * q = p * S.den
  have h_cross : S.num * (q : ℤ) = (p : ℤ) * S.den := by exact_mod_cast h_cross_r
  -- 1979 | S.num * q
  have h1979_numq : (1979 : ℤ) ∣ S.num * (q : ℤ) := sum_num_div_1979.mul_right q
  -- So 1979 | p * S.den
  rw [h_cross] at h1979_numq
  -- 1979 prime
  have h1979_prime : _root_.Prime (1979 : ℤ) := by
    rw [Int.prime_iff_natAbs_prime]; native_decide
  -- 1979 ∤ S.den
  have h_den_ndvd : ¬ (1979 : ℤ) ∣ (S.den : ℤ) := by
    intro h; exact sum_den_not_div_1979 (Int.ofNat_dvd.mp h)
  -- 1979 | p
  exact_mod_cast (h1979_prime.dvd_or_dvd h1979_numq).resolve_right h_den_ndvd
