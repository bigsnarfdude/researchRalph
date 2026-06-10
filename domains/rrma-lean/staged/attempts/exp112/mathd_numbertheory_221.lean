import Mathlib
set_option maxHeartbeats 4000000

theorem mathd_numbertheory_221 (S : Finset ℕ)
  (h₀ : ∀ x : ℕ, x ∈ S ↔ 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) : S.card = 11 := by
  have hS : S = (Finset.Icc 1 999).filter (fun x => x.divisors.card = 3) := by
    ext x; simp only [Finset.mem_filter, Finset.mem_Icc]
    constructor
    · intro hx; obtain ⟨h1, h2, h3⟩ := (h₀ x).mp hx; exact ⟨⟨h1, by omega⟩, by exact_mod_cast h3⟩
    · intro ⟨⟨h1, h2⟩, h3⟩; exact (h₀ x).mpr ⟨h1, by omega, h3⟩
  rw [hS]; native_decide
