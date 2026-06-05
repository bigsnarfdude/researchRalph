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
  -- Best attempt: block construction A = {0,1,2,3} ∪ ⋃ⱼ [4^j, 2·4^j].
  -- The BASIS half is fully proved below. The partition half (part 2) is the
  -- irreducible hard core of Erdős 741(ii): no arithmetic construction tried
  -- survives an "even/odd within each block" refinement (see MISTAKES.md).
  refine ⟨{n | n ≤ 3 ∨ ∃ j, 4 ^ j ≤ n ∧ n ≤ 2 * 4 ^ j}, ?_, ?_⟩
  · intro n hn
    have hn0 : n ≠ 0 := by omega
    have hle : 4 ^ Nat.log 4 n ≤ n := Nat.pow_log_le_self 4 hn0
    have hlt : n < 4 ^ (Nat.log 4 n + 1) := Nat.lt_pow_succ_log_self (by norm_num) n
    rw [pow_succ] at hlt
    set k := Nat.log 4 n with hkdef
    by_cases hc1 : n ≤ 2 * 4 ^ k
    · exact ⟨0, Or.inl (by norm_num), n, Or.inr ⟨k, hle, hc1⟩, by omega⟩
    · by_cases hc2 : n ≤ 3 * 4 ^ k
      · refine ⟨4 ^ k, Or.inr ⟨k, le_refl _, by omega⟩,
          n - 4 ^ k, Or.inr ⟨k, by omega, by omega⟩, by omega⟩
      · refine ⟨n - 2 * 4 ^ k, Or.inr ⟨k, by omega, by omega⟩,
          2 * 4 ^ k, Or.inr ⟨k, by omega, by omega⟩, by omega⟩
  · intro A₁ A₂ h1 h2 hcov hdisj
    sorry

end Erdos741OAI
