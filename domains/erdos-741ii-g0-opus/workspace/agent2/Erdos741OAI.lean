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
  refine ⟨{x | x ≤ 3} ∪ {x | 4 ∣ x}, ?_, ?_⟩
  · -- Part 1: basis of order 2.  n = (n % 4) + (n - n % 4)
    intro n hn
    refine ⟨n % 4, ?_, n - n % 4, ?_, ?_⟩
    · refine Or.inl ?_
      show n % 4 ≤ 3
      omega
    · refine Or.inr ?_
      show 4 ∣ (n - n % 4)
      omega
    · omega
  · -- Part 2: coloring obstruction.
    -- NOTE (honest status): Part 1 (basis of order 2) is fully proved above for
    -- A = {0,1,2,3} ∪ 4ℕ.  Part 2 is the genuine core of Erdős #741(ii) and does
    -- NOT hold for this A: the partition A₁={0,1,2,3}∪{4,12,20,…}, A₂={8,16,24,…}
    -- gives A₁+A₁ ⊇ 8ℕ and A₂+A₂ = 8ℕ≥16, both syndetic.  The same colorability
    -- defect kills every arithmetic-progression-bulk basis.  A correct witness needs
    -- Erdős's non-AP "scale-separated" basis, whose Part-2 proof is a substantial
    -- formalization (forced representations across growing scales).  Left as `sorry`
    -- rather than fabricate; see MISTAKES.md / LEARNINGS.md for the analysis.
    sorry

end Erdos741OAI
