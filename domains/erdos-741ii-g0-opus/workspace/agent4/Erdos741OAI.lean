import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-
  ANALYSIS (agent4) — Erdős #741(ii).

  We need a SINGLE set A that is simultaneously:
    (C1) an additive basis of order 2 for all n ≥ 4, and
    (C2) "partition-irreducible": for EVERY 2-partition A = A₁ ⊔ A₂,
         at least one of A₁+A₁, A₂+A₂ is NOT syndetic (has unbounded gaps).

  C1 forces A to be fairly DENSE: covering every n ≥ 4 by a+b means
  A+A ⊇ [4,∞), which is itself syndetic, and a basis cannot be too sparse
  (|A ∩ [0,n]| ≳ √n). So A cannot be a sparse/Sidon-type set.

  C2 is a strong Ramsey-type property and is the research-hard core.
  Candidate witnesses that FAIL C2 (verified by hand):
    • A = univ            — split evens/odds: evens+evens = evens (gap 2,
                            syndetic), odds+odds = evens (syndetic). Both
                            syndetic ⇒ C2 fails.
    • A = {0} ∪ ⋃ₖ [4ᵏ, 2·4ᵏ]  — C1 holds (0 fills blocks, self-sums
                            [2·4ᵏ,4ᵏ⁺¹] tile the gaps), but C2 fails: color
                            each block by parity ⇒ both color classes keep
                            sumset gaps ≤ 2, both syndetic.
    • any union of INTERVAL blocks — defeated by the even/odd colouring of
                            each block (an interval halves into two APs whose
                            self-sums are still syndetic).

  The obstruction: to defeat ALL colourings (incl. the alternating/parity
  ones), the blocks may not be intervals or APs — a structure where every
  half-density subset loses syndeticity is required. Such a witness is the
  genuine content of Erdős #741(ii) and is not reducible to a short Lean
  construction within this loop. Documented in LEARNINGS/MISTAKES.

  Below: C1 is discharged with the simplest valid basis (univ) to keep the
  basis half machine-checked; C2 remains an open goal — it is FALSE for
  univ, so this is an honest scaffold, NOT a completed proof.  No sound proof
  of C2 for univ exists; closing it requires replacing the witness.
-/
theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨Set.univ, ?_, ?_⟩
  · -- (C1) univ is trivially a basis of order 2: n = 0 + n
    intro n hn
    exact ⟨0, Set.mem_univ 0, n, Set.mem_univ n, by omega⟩
  · -- (C2) OPEN: research-hard, and in fact FALSE for the univ witness.
    sorry

end Erdos741OAI
