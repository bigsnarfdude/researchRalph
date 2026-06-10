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
  refine ⟨{n | ∃ k : ℕ, 4 ^ k ≤ n ∧ n ≤ 2 * 4 ^ k}, ?_, ?_⟩
  · intro n hn
    -- Lacunary basis: each n ≥ 4 is a sum of two block elements.
    set m := n / 2 with hm
    have hm0 : m ≠ 0 := by omega
    have hlo : (4:ℕ) ^ Nat.log 4 m ≤ m := Nat.pow_log_le_self 4 hm0
    have hhi : m < (4:ℕ) ^ (Nat.log 4 m + 1) := Nat.lt_pow_succ_log_self (by norm_num) m
    set j := Nat.log 4 m with hj
    have hp1 : (4:ℕ) ^ (j + 1) = 4 * 4 ^ j := by rw [pow_succ]; ring
    by_cases hc : n ≤ 4 * 4 ^ j
    · exact ⟨n / 2, ⟨j, by omega, by omega⟩, n - n / 2, ⟨j, by omega, by omega⟩, by omega⟩
    · exact ⟨n - 1, ⟨j + 1, by omega, by omega⟩, 1, ⟨0, by norm_num, by norm_num⟩, by omega⟩
  · intro A₁ A₂ h1 h2 hcov hdis
    -- HONEST STATUS: condition 2 is NOT closed and, for THIS construction, is in fact false:
    -- the parity coloring (A₁ = even elements of A, A₂ = odd elements) makes both A₁+A₁ and
    -- A₂+A₂ cover all even numbers with gap 2 (full blocks hold both parities; the small
    -- elements 1,2 bridge inter-block regions). A valid construction must break AP/residue
    -- structure at every scale (the hard core of Erdős 741(ii)); not found this session.
    sorry

end Erdos741OAI
