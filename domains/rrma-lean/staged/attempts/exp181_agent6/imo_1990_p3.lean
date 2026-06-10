import Mathlib
set_option maxHeartbeats 3200000

private lemma imo1990_n_odd {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1) : ¬ 2 ∣ n := by
  intro h2n; have : 2 ∣ 2 ^ n + 1 := dvd_trans (dvd_pow h2n two_ne_zero) hdvd
  have : 2 ∣ 2 ^ n := dvd_pow_self 2 (by omega); omega

private lemma order_arg {n p : ℕ} (hp : Nat.Prime p) (hp2 : p ≠ 2)
    (hp_dvd : p ∣ 2 ^ n + 1) (hn_odd : ¬ 2 ∣ n) :
    ∃ e, e ∣ n ∧ 0 < e ∧ e < p ∧ p ∣ 2 ^ e + 1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hp3 : 3 ≤ p := by have := hp.two_le; omega
  have hndvd2 : ¬ p ∣ 2 := by intro h; exact absurd (Nat.le_of_dvd (by omega) h) (by omega)
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
    exact h2nz (by linear_combination h2n1)
  have hd_even : 2 ∣ d := by
    by_contra h
    exact hdn (Nat.coprime_comm.mpr ((Nat.Prime.coprime_iff_not_dvd Nat.prime_two).mpr h)
      |>.dvd_of_dvd_mul_right (mul_comm 2 n ▸ hd2n))
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
    · exfalso; have : 2 * e ≤ e := Nat.le_of_dvd (by omega) (he ▸ orderOf_dvd_of_pow_eq_one h); omega
    · exact h
  exact ⟨e, he_dvd, he_pos, he_lt, by
    rw [← ZMod.natCast_eq_zero_iff]; push_cast; linear_combination h2e1⟩

private lemma imo1990_minFac {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1) :
    n.minFac = 3 := by
  have h_odd := imo1990_n_odd hn hdvd
  set p := n.minFac
  have hp : Nat.Prime p := Nat.minFac_prime (by omega)
  have hp2 : p ≠ 2 := by
    intro h; exact h_odd (h ▸ Nat.minFac_dvd n)
  obtain ⟨e, he_dvd, _, he_lt, hp_e⟩ :=
    order_arg hp hp2 (dvd_trans (Nat.minFac_dvd n) (dvd_trans (dvd_pow_self n (by omega)) hdvd)) h_odd
  have : e = 1 := by
    by_contra hne; have hq := Nat.minFac_prime (show e ≠ 1 by omega)
    have : e.minFac < p := lt_of_le_of_lt (Nat.minFac_le (by omega)) he_lt
    have : p ≤ e.minFac := Nat.minFac_le_of_dvd hq.two_le (dvd_trans (Nat.minFac_dvd e) he_dvd)
    omega
  rw [this] at hp_e; simp at hp_e
  have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd p hp_e; omega

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  have h_odd := imo1990_n_odd h₀ h₁
  have h3n : 3 ∣ n := imo1990_minFac h₀ h₁ ▸ Nat.minFac_dvd n
  obtain ⟨m, hm⟩ := h3n; subst hm
  have hm_pos : 0 < m := by omega
  -- ¬ 3 | m (via LTE)
  have hm3 : ¬ 3 ∣ m := by
    intro h3m; set nn := 3 * m
    have hn_odd : Odd nn := by rw [Nat.odd_iff]; omega
    have hp3 : Prime (3 : ℕ) := Nat.prime_iff.mp (by norm_num)
    have h_lte := Nat.emultiplicity_pow_add_pow (show Nat.Prime 3 by norm_num)
      (show Odd 3 by decide) (show (3 : ℕ) ∣ 2 + 1 by norm_num)
      (show ¬ (3 : ℕ) ∣ 2 by norm_num) hn_odd
    simp only [one_pow] at h_lte; norm_num at h_lte
    have h_le := emultiplicity_le_emultiplicity_of_dvd_right (a := 3) h₁
    rw [emultiplicity_pow hp3, h_lte,
        show emultiplicity (3 : ℕ) 3 = 1 from
          emultiplicity_eq_of_dvd_of_not_dvd (by norm_num) (by norm_num)] at h_le
    have h_ge : (2 : ℕ∞) ≤ emultiplicity 3 nn := le_emultiplicity_of_pow_dvd (show 3 ^ 2 ∣ nn from by
      show 9 ∣ 3 * m; obtain ⟨k, hk⟩ := h3m; subst hk; omega)
    have h_fin : emultiplicity 3 nn ≠ ⊤ := by
      rw [Ne, emultiplicity_eq_top, not_not]; refine ⟨nn, ?_⟩
      intro h; exact absurd (Nat.le_of_dvd (by omega) h)
        (by have := Nat.lt_pow_self (show 1 < 3 by omega) (n := nn + 1); omega)
    obtain ⟨v, hv⟩ := WithTop.ne_top_iff_exists.mp h_fin
    rw [← hv] at h_le h_ge
    have : 2 ≤ v := by rw [show (2 : ℕ∞) = ↑(2 : ℕ) from rfl] at h_ge; exact WithTop.coe_le_coe.mp h_ge
    have : 2 * v ≤ 1 + v := by
      have : (↑(2 * v) : ℕ∞) ≤ ↑(1 + v) := by push_cast; exact h_le
      exact_mod_cast this
    omega
  -- m = 1
  have hm1 : m = 1 := by
    by_contra hm_ne1; have hm2 : 2 ≤ m := by omega
    set q := m.minFac
    have hq : Nat.Prime q := Nat.minFac_prime (by omega)
    have hq_dvd_m := Nat.minFac_dvd m
    have hq2 : q ≠ 2 := by
      intro h; exact h_odd (dvd_mul_of_dvd_right (h ▸ hq_dvd_m) 3)
    have hq3 : q ≠ 3 := by
      intro h; exact hm3 (h ▸ hq_dvd_m)
    have hq5 : 5 ≤ q := by
      by_contra h; push_neg at h; interval_cases q <;> simp_all (config := { decide := true })
    obtain ⟨e, he_dvd, he_pos, he_lt, hq_e⟩ :=
      order_arg hq hq2 (dvd_trans (dvd_mul_of_dvd_right (Nat.minFac_dvd m) 3)
        (dvd_trans (dvd_pow_self (3*m) (by omega)) h₁)) h_odd
    -- e | 3m, e < q = minFac(m). gcd(e,m) = 1.
    have he_cop : Nat.Coprime e m := by
      rw [Nat.Coprime]; by_contra h
      have h3 := Nat.minFac_prime (show Nat.gcd e m ≠ 1 from h)
      have h5 := Nat.le_of_dvd he_pos (dvd_trans (Nat.minFac_dvd _) (Nat.gcd_dvd_left e m))
      have h7 := Nat.minFac_le_of_dvd h3.two_le (dvd_trans (Nat.minFac_dvd _) (Nat.gcd_dvd_right e m))
      linarith
    have he_dvd_3 : e ∣ 3 := he_cop.dvd_of_dvd_mul_right he_dvd
    -- e ∈ {1, 3}
    have : e = 1 ∨ e = 3 := by
      have := Nat.le_of_dvd (by omega) he_dvd_3; interval_cases e <;> omega
    rcases this with rfl | rfl
    · simp at hq_e; have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd q hq_e; omega
    · simp at hq_e; have : q ∣ 3 := hq.dvd_of_dvd_pow (show q ∣ 3 ^ 2 by omega)
      have := (show Nat.Prime 3 from by norm_num).eq_one_or_self_of_dvd q this; omega
  omega
