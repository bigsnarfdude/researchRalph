import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_405 (S : Finset ℕ) (h₀ : ∀ x, x ∈ S ↔ 0 < x ∧ x ^ 2 + 4 * x + 4 < 20) :
  S.card = 2 := by
  have hS : S = {1, 2} := by
    ext x; rw [h₀]; simp only [Finset.mem_insert, Finset.mem_singleton]
    constructor
    · intro ⟨hx, hlt⟩
      have hxle : x ≤ 3 := by nlinarith
      interval_cases x <;> omega
    · intro h; rcases h with rfl | rfl <;> omega
  rw [hS]; decide
