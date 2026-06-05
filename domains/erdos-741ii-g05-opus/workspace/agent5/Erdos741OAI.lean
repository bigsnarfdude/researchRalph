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
  refine ⟨{n | n = 0 ∨ ∃ k, 1 ≤ k ∧ 4 ^ k ≤ n ∧ n ≤ 2 * 4 ^ k}, ?_, ?_⟩
  · -- Basis of order 2: every n ≥ 4 is a sum of two elements
    intro n hn
    set k := Nat.log 4 n with hk
    have hn0 : n ≠ 0 := by omega
    have hkpos : 0 < k := Nat.log_pos (by norm_num) (by omega)
    have hk1 : 1 ≤ k := hkpos
    have hmle : 4 ^ k ≤ n := Nat.pow_log_le_self 4 hn0
    have hub : n < 4 ^ (k + 1) := Nat.lt_pow_succ_log_self (by norm_num) n
    set m := 4 ^ k with hm
    have hub' : n < 4 * m := by rw [pow_succ] at hub; omega
    have hm4 : 4 ≤ m := by
      have : (4:ℕ) ^ 1 ≤ 4 ^ k := Nat.pow_le_pow_right (by norm_num) hk1
      simpa using this
    rcases (le_or_lt n (2 * m) : n ≤ 2 * m ∨ 2 * m < n) with h2 | h2
    · exact ⟨n, Or.inr ⟨k, hk1, hmle, h2⟩, 0, Or.inl rfl, by omega⟩
    · rcases (le_or_lt n (3 * m) : n ≤ 3 * m ∨ 3 * m < n) with h3 | h3
      · exact ⟨m, Or.inr ⟨k, hk1, by omega, by omega⟩, n - m,
          Or.inr ⟨k, hk1, by omega, by omega⟩, by omega⟩
      · exact ⟨n - 2 * m, Or.inr ⟨k, hk1, by omega, by omega⟩, 2 * m,
          Or.inr ⟨k, hk1, by omega, by omega⟩, by omega⟩
  · -- Partition condition.
    -- NOTE: the interval construction A = {0} ∪ ⋃ₖ [4ᵏ, 2·4ᵏ] proven above is a
    -- valid basis of order 2, but it does NOT satisfy this partition condition:
    -- the mod-4 colouring A₁ = {x ∈ A : x ≡ 0,1 mod 4}, A₂ = {x ∈ A : x ≡ 2,3 mod 4}
    -- makes BOTH A₁+A₁ and A₂+A₂ syndetic (each covers residues {0,1,2} mod 4 with
    -- bounded gaps; the fixed small elements 4,5,6,7 shift-cover every block bottom).
    -- A correct witness needs a thin / self-similar basis whose blocks are themselves
    -- sparse, so that the shift elements grow with the scale and force unbounded gaps.
    sorry

end Erdos741OAI
