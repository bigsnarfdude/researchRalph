import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p3 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ Nat.Prime (n ^ 2 + 2 - 3 * n)) :
  S.card = 1 := by
  convert_to ({3} : Finset ℕ).card = 1
  · congr 1
    ext n
    simp only [Finset.mem_singleton, h₀]
    constructor
    · rintro ⟨hn_pos, hn_prime⟩
      by_contra hne
      have hn1 : n ≠ 1 := by intro h; subst h; norm_num at hn_prime
      have hn2 : n ≠ 2 := by intro h; subst h; norm_num at hn_prime
      have hn4 : n ≥ 4 := by omega
      have hfact : n ^ 2 + 2 - 3 * n = (n - 1) * (n - 2) := by
        have h1 : 3 * n ≤ n ^ 2 + 2 := by nlinarith
        have h2 : 2 ≤ n := by omega
        have h3 : 1 ≤ n := by omega
        zify [h1, h2, h3]; ring
      rw [hfact] at hn_prime
      have h_dvd : (n - 2) ∣ ((n - 1) * (n - 2)) := dvd_mul_left _ _
      rcases hn_prime.eq_one_or_self_of_dvd (n - 2) h_dvd with h | h
      · omega
      · have hne0 : n - 2 ≠ 0 := by omega
        have h_eq : 1 * (n - 2) = (n - 1) * (n - 2) := by linarith
        have := mul_right_cancel₀ hne0 h_eq
        omega
    · rintro rfl
      exact ⟨by norm_num, by norm_num⟩
  · simp
