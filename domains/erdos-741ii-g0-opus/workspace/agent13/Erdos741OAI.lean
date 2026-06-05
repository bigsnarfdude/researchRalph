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
  -- E = bits only at even positions, O = bits only at odd positions; A = E ∪ O.
  -- Every n = e + o with e∈E, o∈O (disjoint bit supports), so A is a basis of order 2.
  classical
  set E : Set ℕ := {n | ∀ i, n.testBit (2 * i + 1) = false} with hE
  set O : Set ℕ := {n | ∀ i, n.testBit (2 * i) = false} with hO
  have basis : ∀ n : ℕ, ∃ e ∈ E, ∃ o ∈ O, e + o = n := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      rcases Nat.eq_zero_or_pos n with hn | hn
      · exact ⟨0, by intro i; simp [hE], 0, by intro i; simp [hO], by simp [hn]⟩
      · obtain ⟨e', he', o', ho', hsum'⟩ := ih (n / 2) (Nat.div_lt_self hn one_lt_two)
        refine ⟨n % 2 + 2 * o', ?_, 2 * e', ?_, ?_⟩
        · intro i
          have hdiv : (n % 2 + 2 * o') / 2 = o' := by omega
          rw [Nat.testBit_succ, hdiv]
          exact ho' i
        · intro i
          rcases Nat.eq_zero_or_pos i with hi | hi
          · subst hi; simp [Nat.testBit_zero]
          · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
            have h2 : 2 * (j + 1) = (2 * j + 1) + 1 := by ring
            have hdiv : (2 * e') / 2 = e' := by omega
            rw [h2, Nat.testBit_succ, hdiv]
            exact he' j
        · have := Nat.div_add_mod n 2
          omega
  refine ⟨E ∪ O, ?_, ?_⟩
  · intro n _
    obtain ⟨e, he, o, ho, hsum⟩ := basis n
    exact ⟨e, Or.inl he, o, Or.inr ho, hsum⟩
  · intro A₁ A₂ _ _ _ _
    -- OPEN / DEAD END for THIS construction: the partition property is FALSE for
    -- A = E ∪ O.  Refutation (see MISTAKES.md cand.3): color E by bit0, O by bit1.
    -- Then every n≡0 mod4 ∈ A₁+A₁ and every n≡3 mod4 ∈ A₂+A₂, so BOTH are syndetic.
    -- Hence this sorry is unfillable; the correct Erdős-741(ii) construction must
    -- couple digit positions (no free low digit) — not reconstructed cold here.
    sorry

end Erdos741OAI
