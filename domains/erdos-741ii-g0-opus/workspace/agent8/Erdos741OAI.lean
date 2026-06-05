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
  -- Erdős #741(ii): requires an "unsplittable" additive basis of order 2.
  -- Part 1 (basis) is easy; Part 2 requires that for EVERY 2-partition, one
  -- self-sumset is non-syndetic (unbounded gaps). Every simple witness (univ,
  -- {0,1}∪evens, {0,1,2}∪3ℕ⁺, interval blocks) is defeated by the even/odd or
  -- AP-splitting coloring, which keeps both self-sumsets syndetic. The genuine
  -- construction is the hard content of the problem and is not yet formalized.
  sorry

end Erdos741OAI
