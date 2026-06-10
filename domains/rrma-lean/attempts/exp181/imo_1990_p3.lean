import Mathlib

set_option maxHeartbeats 12800000

open Nat

-- Step 1: n is odd
private lemma odd_of_sq_dvd {n : ℕ} (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : ¬ 2 ∣ n := by
  intro h2n
  have h2sq : 2 ∣ n ^ 2 := dvd_pow h2n (by omega)
  have h2sum : 2 ∣ 2 ^ n + 1 := dvd_trans h2sq h₁
  have heven_pow : 2 ∣ 2 ^ n := dvd_pow_self 2 (by omega : n ≠ 0)
  have := Nat.dvd_sub h2sum heven_pow
  simp at this

-- Helper: (2 : ZMod p)^n = -1 when p prime, p | n, n² | 2^n+1
private lemma zmod_two_pow_eq_neg_one {n p : ℕ} (hp : Nat.Prime p) (hpn : p ∣ n)
    (h₁ : n ^ 2 ∣ 2 ^ n + 1) : (2 : ZMod p) ^ n = -1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hpdvd : p ∣ 2 ^ n + 1 := dvd_trans (dvd_pow hpn (by omega : 2 ≠ 0)) h₁
  have hcast : ((2 ^ n + 1 : ℕ) : ZMod p) = 0 := by
    rwa [ZMod.natCast_eq_zero_iff]
  push_cast at hcast
  exact eq_neg_of_add_eq_zero_left hcast

-- Helper: orderOf (2 : ZMod p) divides 2n
private lemma ord_dvd_two_mul {n p : ℕ} (hp : Nat.Prime p) (hpn : p ∣ n)
    (h₁ : n ^ 2 ∣ 2 ^ n + 1) :
    orderOf (2 : ZMod p) ∣ 2 * n := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have h := zmod_two_pow_eq_neg_one hp hpn h₁
  apply orderOf_dvd_of_pow_eq_one
  have : ((2 : ZMod p) ^ n) ^ 2 = 1 := by rw [h]; ring
  rw [← pow_mul] at this
  exact this

-- Helper: orderOf (2 : ZMod p) does NOT divide n (when p ≥ 3)
private lemma ord_not_dvd {n p : ℕ} (hp : Nat.Prime p) (hpn : p ∣ n)
    (h₁ : n ^ 2 ∣ 2 ^ n + 1) (hp3 : p ≥ 3) :
    ¬ orderOf (2 : ZMod p) ∣ n := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  intro hord
  have h := zmod_two_pow_eq_neg_one hp hpn h₁
  have h1 : (2 : ZMod p) ^ n = 1 := by
    rwa [orderOf_dvd_iff_pow_eq_one] at hord
  -- So -1 = 1 in ZMod p, but p ≥ 3
  have : (-1 : ZMod p) = 1 := by rw [← h]; exact h1
  haveI : Fact (2 < p) := ⟨by omega⟩
  exact ZMod.neg_one_ne_one this

-- Key: if orderOf divides 2n but not n, and n is odd, then 2 | orderOf
private lemma two_dvd_ord {n p : ℕ} (hp : Nat.Prime p) (hpn : p ∣ n)
    (h₁ : n ^ 2 ∣ 2 ^ n + 1) (hp3 : p ≥ 3) (hodd : ¬ 2 ∣ n) :
    2 ∣ orderOf (2 : ZMod p) := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hdvd2n := ord_dvd_two_mul hp hpn h₁
  have hndvd := ord_not_dvd hp hpn h₁ hp3
  -- orderOf | 2n, orderOf ∤ n, n odd → 2 | orderOf
  by_contra h2e
  -- If ¬(2 | orderOf), then orderOf is odd
  -- orderOf | 2n and orderOf is odd and gcd(orderOf, 2) = 1 → orderOf | n
  have hcop : Nat.Coprime (orderOf (2 : ZMod p)) 2 := by
    rwa [Nat.Prime.coprime_iff_not_dvd (by norm_num : Nat.Prime 2)]
  have := hcop.dvd_of_dvd_mul_left hdvd2n
  exact hndvd this

-- The core order argument: if p = minFac(n), then orderOf = 2, hence p | 3
-- Key insight: orderOf = 2f with f | n and f < p = minFac(n), so f = 1
private lemma orderOf_eq_two_of_minFac {n : ℕ} (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1)
    (hodd : ¬ 2 ∣ n) : orderOf (2 : ZMod (n.minFac)) = 2 := by
  set p := n.minFac
  have hn1 : n ≠ 1 := by omega
  have hp : Nat.Prime p := Nat.minFac_prime hn1
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hpn : p ∣ n := Nat.minFac_dvd n
  have hp3 : p ≥ 3 := by
    by_contra h
    push_neg at h
    interval_cases p
    · exact (Nat.Prime.one_lt hp).ne' rfl
    · exact (Nat.Prime.one_lt hp).ne' rfl
    · exact hodd (dvd_trans (Nat.minFac_dvd n) (dvd_refl n) |> sorry)
  sorry

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  have hodd := odd_of_sq_dvd h₀ h₁
  sorry
