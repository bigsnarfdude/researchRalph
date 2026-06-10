import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_303 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 2 ≤ n ∧ 171 ≡ 80 [MOD n] ∧ 468 ≡ 13 [MOD n]) : (∑ k ∈ S, k) = 111 := by
  have hS : S = {7, 13, 91} := by
    ext n
    simp only [Finset.mem_insert, Finset.mem_singleton, h₀]
    constructor
    · rintro ⟨hn2, hmod1, hmod2⟩
      have h91 : n ∣ 91 := by
        have hmod1' := hmod1.symm
        rwa [Nat.modEq_iff_dvd' (by omega)] at hmod1'
      have hle : n ≤ 91 := Nat.le_of_dvd (by omega) h91
      interval_cases n <;> omega
    · rintro (rfl | rfl | rfl) <;> refine ⟨by omega, ?_, ?_⟩ <;> decide
  rw [hS]; decide
