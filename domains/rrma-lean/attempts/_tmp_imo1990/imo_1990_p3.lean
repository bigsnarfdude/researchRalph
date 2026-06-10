import Mathlib

set_option maxHeartbeats 12800000

open Nat

private lemma odd_of_sq_dvd {n : ℕ} (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : ¬ 2 ∣ n := by
  intro h2n
  have h2sq : 2 ∣ n ^ 2 := dvd_pow h2n (by omega)
  have h2sum : 2 ∣ 2 ^ n + 1 := dvd_trans h2sq h₁
  have heven_pow : 2 ∣ 2 ^ n := dvd_pow_self 2 (by omega : n ≠ 0)
  have hsub : 2 ^ n + 1 - 2 ^ n = 1 := by omega
  have := Nat.dvd_sub' h2sum heven_pow
  rw [hsub] at this
  omega

private lemma is_pow_three {n : ℕ} (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1)
    (hodd : ¬ 2 ∣ n) : ∃ k : ℕ, 0 < k ∧ n = 3 ^ k := by
  sorry

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  have hodd := odd_of_sq_dvd h₀ h₁
  obtain ⟨k, hk, rfl⟩ := is_pow_three h₀ h₁ hodd
  -- n = 3^k, k ≥ 1. Show k = 1 via LTE.
  have hlte := @Nat.emultiplicity_pow_add_pow 3 (by norm_num) (by norm_num)
    (x := 2) (y := 1) (by norm_num) (by norm_num) (n := 3 ^ k)
    (Odd.pow (by norm_num : Odd 3))
  simp only [one_pow] at hlte
  rw [Nat.Prime.emultiplicity_self (by norm_num),
      Nat.Prime.emultiplicity_pow_self (by norm_num)] at hlte
  -- hlte : emultiplicity 3 (2 ^ 3 ^ k + 1) = 1 + ↑k
  have hdvd : 3 ^ (2 * k) ∣ 2 ^ 3 ^ k + 1 := by
    have : (3 ^ k) ^ 2 = 3 ^ (2 * k) := by ring
    rwa [← this]
  have hge : ↑(2 * k) ≤ emultiplicity 3 (2 ^ 3 ^ k + 1) :=
    le_emultiplicity_of_pow_dvd hdvd
  rw [hlte] at hge
  -- hge : ↑(2 * k) ≤ 1 + ↑k  (in ℕ∞)
  norm_cast at hge
  -- hge should now be : 2 * k ≤ 1 + k  (in ℕ)
  -- k ≤ 1 and k ≥ 1 → k = 1
  clear h₁ h₀ hodd hlte hdvd
  omega
