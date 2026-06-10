import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_221 (S : Finset ℕ)
  (h₀ : ∀ x : ℕ, x ∈ S ↔ 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) : S.card = 11 := by
  have hS : S = Finset.filter (fun x => 0 < x ∧ x < 1000 ∧ x.divisors.card = 3) (Finset.range 1000) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · intro hx; have := (h₀ x).mp hx; exact ⟨by omega, this⟩
    · intro ⟨hx, h1, h2, h3⟩; exact (h₀ x).mpr ⟨h1, h2, h3⟩
  rw [hS]
  native_decide
