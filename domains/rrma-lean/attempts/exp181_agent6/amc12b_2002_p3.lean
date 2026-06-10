import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem amc12b_2002_p3 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ Nat.Prime (n ^ 2 + 2 - 3 * n)) :
  S.card = 1 := by
  have hS : S = {3} := by
    ext n; simp only [Finset.mem_singleton, h₀]
    constructor
    · intro ⟨hn_pos, hn_prime⟩
      have hn_le : n ≤ 3 := by
        by_contra h; push_neg at h
        have hfact : n ^ 2 + 2 - 3 * n = (n - 1) * (n - 2) := by
          zify [show 1 ≤ n by omega, show 2 ≤ n by omega, show 3 * n ≤ n ^ 2 + 2 by nlinarith]; ring
        rw [hfact] at hn_prime
        have h_dvd : (n - 2) ∣ (n - 1) * (n - 2) := dvd_mul_left (n - 2) (n - 1)
        rcases hn_prime.eq_one_or_self_of_dvd (n - 2) h_dvd with h1 | h2
        · omega
        · have h2' : 1 * (n - 2) = (n - 1) * (n - 2) := by rw [one_mul]; exact h2
          have := Nat.eq_of_mul_eq_mul_right (by omega : 0 < n - 2) h2'; omega
      interval_cases n
      · exact absurd hn_prime (by decide)
      · exact absurd hn_prime (by decide)
      · rfl
    · intro hn; subst hn; exact ⟨by norm_num, by norm_num⟩
  rw [hS, Finset.card_singleton]
