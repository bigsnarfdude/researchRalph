import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Candidate 1: geometric scales Q k = 3^k.
At scale k: filler F k = [2·Q k, 3·Q k), rigid point p k = 4·Q k, block B k = [5·Q k, 6·Q k].
Test interval J k = p k + B k = [9·Q k, 10·Q k). -/
def Qs (k : ℕ) : ℕ := 3 ^ k

def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k : ℕ, (Icc (2 * Qs k) (3 * Qs k - 1) ∪ {4 * Qs k} ∪ Icc (5 * Qs k) (6 * Qs k))

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  sorry

end Erdos741OAI
