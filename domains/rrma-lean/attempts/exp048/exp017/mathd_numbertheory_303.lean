import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_303 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 2 ≤ n ∧ 171 ≡ 80 [MOD n] ∧ 468 ≡ 13 [MOD n]) : (∑ k ∈ S, k) = 111 := by
  have hS : S = {7, 13, 91} := by
    ext n; simp only [Finset.mem_insert, Finset.mem_singleton, h₀]
    constructor
    · intro ⟨hn2, h1, h2⟩
      have h_div91 : n ∣ 91 := by
        have h1' : 80 ≡ 171 [MOD n] := h1.symm
        rwa [show (91 : ℕ) = 171 - 80 from by norm_num, ← Nat.modEq_iff_dvd' (by omega : 80 ≤ 171)]
      have hn_le : n ≤ 91 := Nat.le_of_dvd (by norm_num) h_div91
      interval_cases n <;> omega
    · intro h
      rcases h with rfl | rfl | rfl <;> refine ⟨by omega, ?_, ?_⟩ <;> decide
  rw [hS]; decide
