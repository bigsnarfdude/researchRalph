import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- The base-3 "no digit 2" set: smallest set containing 0 and closed under
    x ↦ 3x and x ↦ 3x+1.  Equivalently numbers whose base-3 digits are all 0/1. -/
inductive InA : ℕ → Prop
  | zero : InA 0
  | d0 : ∀ {x}, InA x → InA (3 * x)
  | d1 : ∀ {x}, InA x → InA (3 * x + 1)

/-- Basis of order 2 (in fact for every n): every n = a+b with a,b ∈ A. -/
lemma basisAux : ∀ n : ℕ, ∃ a, InA a ∧ ∃ b, InA b ∧ a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · exact ⟨0, InA.zero, 0, InA.zero, by simp [hn]⟩
    · have hq : n / 3 < n := Nat.div_lt_self hn (by norm_num)
      obtain ⟨a', ha', b', hb', hab'⟩ := ih (n / 3) hq
      have hmod : 3 * (n / 3) + n % 3 = n := Nat.div_add_mod n 3
      have key : n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2 := by omega
      rcases key with h | h | h
      · exact ⟨3 * a', InA.d0 ha', 3 * b', InA.d0 hb', by omega⟩
      · exact ⟨3 * a' + 1, InA.d1 ha', 3 * b', InA.d0 hb', by omega⟩
      · exact ⟨3 * a' + 1, InA.d1 ha', 3 * b' + 1, InA.d1 hb', by omega⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨{n | InA n}, ?_, ?_⟩
  · intro n _hn
    obtain ⟨a, ha, b, hb, hab⟩ := basisAux n
    exact ⟨a, ha, b, hb, hab⟩
  · -- RIGIDITY HALF — UNSOLVED for this construction.
    -- The base-3 set is NOT rigid: the last-digit coloring A₁=A∩3ℕ, A₂=A∩(3ℕ+1)
    -- gives A₁+A₁=3ℕ and A₂+A₂=3ℕ+2, both syndetic. So this `A` does NOT satisfy
    -- the second conjunct. A correct construction must defeat BOTH parity colorings
    -- (kills interval bases) AND residue/digit colorings (kills digit-closed bases),
    -- and rigidity provably needs a global counting argument, not local forcing.
    -- See LEARNINGS.md (agent4 findings) for the full elimination.
    intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    sorry

end Erdos741OAI
