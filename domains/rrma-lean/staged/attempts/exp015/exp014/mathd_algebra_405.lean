import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_405 (S : Finset ℕ) (h₀ : ∀ x, x ∈ S ↔ 0 < x ∧ x ^ 2 + 4 * x + 4 < 20) :
  S.card = 2 := by
  -- x^2 + 4x + 4 = (x+2)^2 < 20, with x > 0
  -- (x+2)^2 < 20 means x+2 < 5 (for positive x), so x < 3, i.e. x ∈ {1, 2}
  have hS : S = {1, 2} := by
    ext x
    simp only [h₀, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · intro ⟨hpos, hlt⟩
      have : x < 3 := by nlinarith [sq_nonneg (x + 2)]
      omega
    · intro h
      rcases h with rfl | rfl <;> norm_num
  rw [hS]
  simp [Finset.card_insert_of_not_mem]