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
  -- Candidate 6: A = {0} ∪ ⋃_k [4^k, 2·4^k]  (doubling blocks; aperiodic, growing gaps).
  -- Block k self-sum = [2·4^k, 4^{k+1}] exactly fills the gap after block k, so this IS a
  -- provable basis.  Rigidity is the hard crux (see MISTAKES) — left as sorry.
  refine ⟨{n | n = 0 ∨ ∃ k, 4 ^ k ≤ n ∧ n ≤ 2 * 4 ^ k}, ?_, ?_⟩
  · intro n hn
    have hn0 : n ≠ 0 := by omega
    obtain ⟨k, h1, h2⟩ : ∃ k, 4 ^ k ≤ n ∧ n < 4 ^ k * 4 := by
      refine ⟨Nat.log 4 n, Nat.pow_log_le_self 4 hn0, ?_⟩
      have := Nat.lt_pow_succ_log_self (b := 4) (by norm_num) n
      rwa [pow_succ] at this
    rcases le_total n (2 * 4 ^ k) with hle | hgt
    · exact ⟨0, Or.inl rfl, n, Or.inr ⟨k, h1, hle⟩, by omega⟩
    · rcases le_total n (3 * 4 ^ k) with hle3 | hgt3
      · exact ⟨4 ^ k, Or.inr ⟨k, le_refl _, by omega⟩,
              n - 4 ^ k, Or.inr ⟨k, by omega, by omega⟩, by omega⟩
      · exact ⟨n - 2 * 4 ^ k, Or.inr ⟨k, by omega, by omega⟩,
              2 * 4 ^ k, Or.inr ⟨k, by omega, by omega⟩, by omega⟩
  · intro A₁ A₂ _ _ _ _
    sorry

end Erdos741OAI
