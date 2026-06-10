import Mathlib

/-!
Erdős Problem #741(ii)
Is there a basis A of order 2 such that if A = A₁ ⊔ A₂ then
A₁+A₁ and A₂+A₂ cannot both be syndetic (bounded gaps)?
Answer: Yes.
-/

open Set Filter

def IsSyndetic' (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ n : ℕ, ∃ m ∈ S, n ≤ m ∧ m ≤ n + C

def IsAddBasis2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ (A ∪ {0}), ∃ b ∈ (A ∪ {0}), a + b = n

def sumset' (S : Set ℕ) : Set ℕ := {n | ∃ a ∈ S, ∃ b ∈ S, a + b = n}

theorem erdos_741_ii :
    ∃ A : Set ℕ,
      IsAddBasis2 A ∧
      ∀ A₁ A₂ : Set ℕ, A = A₁ ∪ A₂ → Disjoint A₁ A₂ →
        ¬(IsSyndetic' (sumset' A₁) ∧ IsSyndetic' (sumset' A₂)) := by
  sorry
