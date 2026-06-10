import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  use {n : ℕ | n ≤ 3 ∨ Even n}
  constructor
  · -- Basis property: every n ≥ 4 is a sum of two elements from A
    intro n hn
    by_cases h : Even n
    · obtain ⟨k, hk⟩ := h
      subst hk
      use 2
      constructor
      · left; norm_num
      · use 2 * (k - 1)
        constructor
        · right; norm_num
        · omega
    · have h_odd : Odd n := Nat.odd_iff_not_even.mpr h
      obtain ⟨k, hk⟩ := h_odd
      subst hk
      use 1
      constructor
      · left; norm_num
      · use 2 * k
        constructor
        · right; norm_num
        · ring
  · -- Partition property: for any partition, at least one sumset is not syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨hsyn1, hsyn2⟩
    by_cases hA₁_empty : A₁ = ∅
    · subst hA₁_empty
      simp [Set.add_empty] at hsyn1
    · by_cases hA₂_empty : A₂ = ∅
      · subst hA₂_empty
        simp [Set.add_empty] at hsyn2
      · have h_zero_in_A : 0 ∈ ({n : ℕ | n ≤ 3 ∨ Even n} : Set ℕ) := by left; norm_num
        by_cases h0A₁ : 0 ∈ A₁
        · exfalso
          sorry
        · exfalso
          sorry

end Erdos741OAI
