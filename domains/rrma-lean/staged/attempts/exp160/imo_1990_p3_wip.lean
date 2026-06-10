import Mathlib

set_option maxHeartbeats 3200000

private lemma imo1990_n_odd {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1) : ¬ 2 ∣ n := by
  intro h2n
  have : 2 ∣ 2 ^ n + 1 := dvd_trans (dvd_pow h2n two_ne_zero) hdvd
  have : 2 ∣ 2 ^ n := dvd_pow_self 2 (by omega)
  omega

-- Core order argument
private lemma order_arg {n p : ℕ} (hp : Nat.Prime p) (hp2 : p ≠ 2)
    (hp_dvd : p ∣ 2 ^ n + 1) (hn_odd : ¬ 2 ∣ n) :
    ∃ e, e ∣ n ∧ 0 < e ∧ e < p ∧ p ∣ 2 ^ e + 1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hp3 : 3 ≤ p := by have := hp.two_le; omega
  have hndvd2 : ¬ p ∣ 2 := by intro h; have := Nat.le_of_dvd (by omega) h; omega
  have h2nz : (2 : ZMod p) ≠ 0 := by
    intro h; apply hndvd2
    have : ((2 : ℕ) : ZMod p) = 0 := by push_cast at h ⊢; exact h
    rwa [ZMod.natCast_eq_zero_iff] at this
  have h2n1 : (2 : ZMod p) ^ n = -1 := by
    have h0 : ((2 ^ n + 1 : ℕ) : ZMod p) = 0 := (ZMod.natCast_eq_zero_iff _ _).mpr hp_dvd
    have h1 : (2 : ZMod p) ^ n + 1 = 0 := by push_cast at h0; exact h0
    linear_combination h1
  have h22n : (2 : ZMod p) ^ (2 * n) = 1 := by
    calc (2 : ZMod p) ^ (2 * n) = ((2 : ZMod p) ^ n) ^ 2 := by ring
    _ = (-1) ^ 2 := by rw [h2n1]
    _ = 1 := by ring
  set d := orderOf (2 : ZMod p)
  have hd2n : d ∣ 2 * n := orderOf_dvd_of_pow_eq_one h22n
  have hdn : ¬ d ∣ n := by
    intro h; rw [orderOf_dvd_iff_pow_eq_one.mp h] at h2n1
    have : (2 : ZMod p) = 0 := by linear_combination (1 : ZMod p) - h2n1
    exact h2nz this
  have hd_even : 2 ∣ d := by
    by_contra h
    exact hdn (Nat.coprime_comm.mpr ((Nat.Prime.coprime_iff_not_dvd Nat.prime_two).mpr h) |>.dvd_of_dvd_mul_right (mul_comm 2 n ▸ hd2n))
  obtain ⟨e, he⟩ := hd_even
  have hd_pos : 0 < d := Nat.pos_of_ne_zero (fun h0 => by simp [h0] at hd2n; omega)
  have he_pos : 0 < e := by omega
  have he_dvd : e ∣ n := (Nat.mul_dvd_mul_iff_left (by omega : 0 < 2)).mp (he ▸ hd2n)
  have hd_pm1 : d ∣ p - 1 := orderOf_dvd_of_pow_eq_one (ZMod.pow_card_sub_one_eq_one h2nz)
  have he_lt : e < p := by have := Nat.le_of_dvd (by omega) hd_pm1; omega
  have h2e1 : (2 : ZMod p) ^ e = -1 := by
    haveI : IsDomain (ZMod p) := ZMod.instIsDomain p
    have hsq : ((2 : ZMod p) ^ e) ^ 2 = 1 := by
      calc ((2 : ZMod p) ^ e) ^ 2 = (2 : ZMod p) ^ (2 * e) := by ring
      _ = (2 : ZMod p) ^ d := by rw [he]
      _ = 1 := pow_orderOf_eq_one _
    rcases sq_eq_one_iff.mp hsq with h | h
    · exfalso; have : d ∣ e := orderOf_dvd_of_pow_eq_one h
      have : 2 * e ∣ e := he ▸ this; omega
    · exact h
  have hp_dvd_2e : p ∣ 2 ^ e + 1 := by
    rw [← ZMod.natCast_eq_zero_iff]
    have : (2 : ZMod p) ^ e + 1 = 0 := by linear_combination h2e1
    push_cast at this ⊢; exact this
  exact ⟨e, he_dvd, he_pos, he_lt, hp_dvd_2e⟩

-- minFac(n) = 3
private lemma imo1990_minFac {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1) :
    n.minFac = 3 := by
  have h_odd := imo1990_n_odd hn hdvd
  set p := n.minFac
  have hp : Nat.Prime p := Nat.minFac_prime (by omega)
  have hp2 : p ≠ 2 := fun h => by rw [h] at *; exact h_odd (Nat.minFac_dvd n)
  have hn_dvd : n ∣ 2 ^ n + 1 := dvd_trans (dvd_pow_self n (by omega)) hdvd
  obtain ⟨e, he_dvd, _, he_lt, hp_e⟩ := order_arg hp hp2 (dvd_trans (Nat.minFac_dvd n) hn_dvd) h_odd
  have he1 : e = 1 := by
    by_contra hne
    have hq := Nat.minFac_prime (show e ≠ 1 by omega)
    have : e.minFac < p := lt_of_le_of_lt (Nat.minFac_le (by omega)) he_lt
    have : p ≤ e.minFac := Nat.minFac_le_of_dvd hq.two_le (dvd_trans (Nat.minFac_dvd e) he_dvd)
    omega
  rw [he1] at hp_e; simp at hp_e
  have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd p hp_e; omega

-- Main theorem
theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  have h_odd := imo1990_n_odd h₀ h₁
  have h_mf := imo1990_minFac h₀ h₁
  have h3n : 3 ∣ n := h_mf ▸ Nat.minFac_dvd n
  obtain ⟨m, hm⟩ := h3n
  subst hm
  -- Show ¬ 3 | m via LTE
  have hm3 : ¬ 3 ∣ m := by
    intro h3m
    -- 9 | n = 3m, so n² has v₃ ≥ 4. But 2^n+1 has v₃ = 1 + v₃(n) ≤ 1 + v₃(n).
    -- More precisely: v₃(n²) = 2*v₃(n) and v₃(2^n+1) = 1+v₃(n) by LTE.
    -- So v₃(n) ≤ 1. But 9|n gives v₃(n) ≥ 2. Contradiction.
    have hn_odd : Odd (3 * m) := by rwa [Nat.odd_iff, Nat.not_even_iff_odd] at h_odd
    have h_lte := Nat.emultiplicity_pow_add_pow (show Nat.Prime 3 by norm_num)
      (show Odd 3 by norm_num) (show (3 : ℕ) ∣ 2 + 1 by norm_num)
      (show ¬ (3 : ℕ) ∣ 2 by norm_num) hn_odd
    -- h_lte : emultiplicity 3 (2^(3m) + 1^(3m)) = emultiplicity 3 3 + emultiplicity 3 (3m)
    simp only [one_pow] at h_lte
    have h_dvd_em := emultiplicity_le_emultiplicity_of_dvd h₁
    rw [emultiplicity_pow] at h_dvd_em
    -- h_dvd_em : 2 * emultiplicity 3 (3m) ≤ emultiplicity 3 (2^(3m)+1)
    rw [h_lte] at h_dvd_em
    -- 2 * emultiplicity 3 (3m) ≤ emultiplicity 3 3 + emultiplicity 3 (3m)
    -- emultiplicity 3 (3m) ≤ emultiplicity 3 3 = 1
    -- But 9 | 3m, so emultiplicity 3 (3m) ≥ 2
    have h9 : 9 ∣ 3 * m := by exact ⟨m / 3, by omega⟩
    sorry
  -- Show m = 1
  have hm_pos : 0 < m := by nlinarith
  have hm1 : m = 1 := by
    by_contra hm_ne1
    set q := m.minFac
    have hq : Nat.Prime q := Nat.minFac_prime (by omega)
    have hq_dvd_m : q ∣ m := Nat.minFac_dvd m
    have hq2 : q ≠ 2 := by
      intro h; rw [h] at hq_dvd_m; exact h_odd ⟨3 * (m / 2), by omega⟩
    have hq3 : q ≠ 3 := by intro h; rw [h] at hq_dvd_m; exact hm3 hq_dvd_m
    have hq5 : 5 ≤ q := by have := hq.two_le; omega
    have hq_dvd_n : q ∣ 3 * m := dvd_mul_of_dvd_right hq_dvd_m 3
    have hn_dvd : 3 * m ∣ 2 ^ (3 * m) + 1 := dvd_trans (dvd_pow_self (3*m) (by omega)) h₁
    obtain ⟨e, he_dvd, he_pos, he_lt, hq_e⟩ :=
      order_arg hq hq2 (dvd_trans hq_dvd_n hn_dvd) h_odd
    -- e | 3m, e < q. All prime factors of e < q = minFac(m) so don't divide m.
    -- Hence gcd(e,m)=1 and e | 3.
    have he_dvd_3 : e ∣ 3 := by
      have he_cop : Nat.Coprime e m := by
        rw [Nat.coprime_comm]
        apply Nat.Coprime.coprime_dvd_right (Nat.minFac_dvd m |> fun _ => ?_)
        sorry -- need coprimality argument
      exact he_cop.dvd_of_dvd_mul_right he_dvd
    -- e ∈ {1, 3}
    interval_cases e
    · -- e = 1: q | 3, q = 3. But q ≥ 5.
      simp at hq_e; have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd q hq_e; omega
    · -- e = 3: q | 9 = 3². q prime → q | 3. q = 3. But q ≥ 5.
      simp at hq_e; have : q ∣ 3 := hq.dvd_of_dvd_pow (show q ∣ 3 ^ 2 by omega)
      have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd q this; omega
  omega
