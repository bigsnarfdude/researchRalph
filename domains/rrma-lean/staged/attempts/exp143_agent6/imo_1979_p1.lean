import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat Finset

-- The sum over ZMod 1979 is 0
private lemma sum_eq_zero_zmod :
    (∑ k ∈ Finset.Icc (1 : ℕ) 1319,
      (-1 : ZMod 1979) ^ (k + 1) * ((k : ZMod 1979)⁻¹)) = 0 := by
  native_decide

-- 1979 is prime
private lemma prime_1979 : Nat.Prime 1979 := by decide

theorem imo_1979_p1 (p q : ℕ) (h₀ : 0 < q)
  (h₁ : (∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1) ^ (k + 1) * ((1 : ℝ) / k)) = p / q) : 1979 ∣ p := by
  -- Work over ℚ. Define the rational sum.
  set S_Q : ℚ := ∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℚ) ^ (k + 1) * (1 / k) with hSQ_def
  -- The ℝ sum equals the cast of S_Q
  have hcast : (S_Q : ℝ) = ∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1 : ℝ) ^ (k + 1) * (1 / k) := by
    push_cast [hSQ_def]
    congr 1; ext k; ring
  -- From h₁: p/q = S_Q in ℝ
  rw [← hcast] at h₁
  -- So p = q * S_Q in ℝ, hence (p : ℚ) = (q : ℚ) * S_Q
  have hpq_Q : (p : ℚ) = q * S_Q := by
    have hq_pos : (0 : ℝ) < q := Nat.cast_pos.mpr h₀
    have := h₁
    rw [eq_div_iff hq_pos.ne'] at this
    have hpq_R : (p : ℝ) = q * (S_Q : ℝ) := by linarith
    exact_mod_cast hpq_R
  -- S_Q = S_Q.num / S_Q.den
  -- 1979 ∤ S_Q.den (since S_Q.den | lcm of {1,...,1319} and 1979 > 1319)
  -- The ZMod computation shows 1979 | S_Q.num
  -- Therefore from p * S_Q.den = q * S_Q.num: 1979 | p

  -- S_Q.den is coprime to 1979
  have hden_coprime : Nat.Coprime S_Q.den 1979 := by
    rw [Nat.Coprime, Nat.coprime_comm]
    apply Nat.Coprime.coprime_dvd_right S_Q.den_dvd
    -- S_Q.den divides the product of denominators, which divides 1319!
    -- Since 1979 is prime > 1319, gcd(1979, 1319!) = 1
    sorry
  -- 1979 | S_Q.num
  have hnum_dvd : (1979 : ℤ) ∣ S_Q.num := by
    sorry
  -- Conclude 1979 | p
  rw [show (p : ℚ) = q * S_Q from hpq_Q] at *
  sorry
